from app.core.pg_client import postgres_client
from app.core.redis_client import redis_client


def get_redis_client():
    return redis_client


def get_postgres_client():
    return postgres_client
