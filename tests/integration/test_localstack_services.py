"""Connectivity coverage for the Compose support services (FR12 audit store).

``docker-compose.yml`` declares two backing services the application reads at
runtime and no test previously touched: ``audit-db`` (PostgreSQL 15, the
immutable audit-trail store) and ``s3-mock`` (LocalStack's S3, the endpoint the
presigned-URL document service targets). Together with the root ``conftest.py``
suite options they existed only as configuration.

This module is what backs them:

* the session-scoped ``localstack`` fixture declared in the root ``conftest.py``
  is consumed directly, so the ``--localstack`` flag decides whether these tests
  attempt a connection at all - the flag is opt-in because the services only
  exist when the Compose stack is up;
* every test skips, never fails, when the service it needs is absent: without
  the flag, without the client library, or when the endpoint cannot be reached.

No mock stands in for either service. The audit store check is a real
``SELECT 1`` over a real socket, and the S3 check is a real ``PUT`` followed by a
real ``HEAD`` against the endpoint, so a passing run means the services answer,
not that a double returned a canned reply.

Endpoints default to the Compose service names and are overridable through the
same environment variables the application itself reads.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

pytestmark = pytest.mark.integration

AUDIT_DB_DSN = (
    os.getenv("AUDIT_DB_DSN")
    or os.getenv("DATABASE_URL")
    or "postgresql://coa:coa_local_only@audit-db:5432/coa_audit"
)
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://s3-mock:4566")
S3_REGION = os.getenv("S3_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "coa-documents")

CONNECT_TIMEOUT_SECONDS = 5


def _require_localstack(localstack: bool) -> None:
    if not localstack:
        pytest.skip("compose support services are opt-in: rerun with --localstack")


def _connect_to_audit_db(localstack: bool) -> Any:
    _require_localstack(localstack)
    psycopg2 = pytest.importorskip("psycopg2", reason="psycopg2 is required to reach audit-db")
    try:
        return psycopg2.connect(AUDIT_DB_DSN, connect_timeout=CONNECT_TIMEOUT_SECONDS)
    except psycopg2.OperationalError as exc:
        pytest.skip(f"audit-db is not reachable: {exc}")


def _s3_client(localstack: bool) -> Any:
    _require_localstack(localstack)
    boto3 = pytest.importorskip("boto3", reason="boto3 is required to reach s3-mock")
    botocore_exceptions = pytest.importorskip("botocore.exceptions")

    client = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        region_name=S3_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
    )
    try:
        client.list_buckets()
    except botocore_exceptions.BotoCoreError as exc:
        pytest.skip(f"s3-mock is not reachable at {S3_ENDPOINT_URL}: {exc}")
    return client


def _ensure_bucket(client: Any, botocore_exceptions: Any, bucket: str) -> None:
    try:
        client.head_bucket(Bucket=bucket)
        return
    except botocore_exceptions.ClientError as exc:
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status not in (403, 404):
            raise

    arguments: dict[str, Any] = {"Bucket": bucket}
    if S3_REGION != "us-east-1":
        arguments["CreateBucketConfiguration"] = {"LocationConstraint": S3_REGION}
    client.create_bucket(**arguments)


class TestAuditDbService:
    """The audit trail store answers a query over a real connection."""

    def test_the_audit_store_answers_a_trivial_query(self, localstack: bool) -> None:
        connection = _connect_to_audit_db(localstack)
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                assert cursor.fetchone() == (1,)
        finally:
            connection.close()

    def test_the_audit_store_reports_the_database_it_opened(self, localstack: bool) -> None:
        connection = _connect_to_audit_db(localstack)
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT current_database(), current_user")
                database, user = cursor.fetchone()
        finally:
            connection.close()

        assert database
        assert user


class TestS3MockService:
    """The S3 endpoint stores and reports back an object."""

    def test_a_put_is_visible_to_a_following_head(self, localstack: bool) -> None:
        client = _s3_client(localstack)
        botocore_exceptions = pytest.importorskip("botocore.exceptions")
        _ensure_bucket(client, botocore_exceptions, S3_BUCKET)

        payload = b"ventura-localstack-connectivity-probe\n"
        key = f"localstack-probe/{uuid.uuid4().hex}.txt"
        client.put_object(Bucket=S3_BUCKET, Key=key, Body=payload, ContentType="text/plain")
        try:
            head = client.head_object(Bucket=S3_BUCKET, Key=key)

            assert head["ContentLength"] == len(payload)
            assert head["ContentType"] == "text/plain"
            assert head["ETag"]
        finally:
            client.delete_object(Bucket=S3_BUCKET, Key=key)

    def test_a_missing_object_is_reported_as_missing(self, localstack: bool) -> None:
        client = _s3_client(localstack)
        botocore_exceptions = pytest.importorskip("botocore.exceptions")
        _ensure_bucket(client, botocore_exceptions, S3_BUCKET)

        key = f"localstack-probe/{uuid.uuid4().hex}.txt"
        with pytest.raises(botocore_exceptions.ClientError) as excinfo:
            client.head_object(Bucket=S3_BUCKET, Key=key)

        assert excinfo.value.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}