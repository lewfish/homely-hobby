import os
from urllib.parse import urlparse
import subprocess


class NotReadableError(Exception):
    pass


class NotWritableError(Exception):
    pass


def get_session():
    # Lazily load boto
    import boto3
    return boto3.Session()


def copy_from(uri: str, path: str) -> None:
    import botocore

    s3 = get_session().client('s3')

    parsed_uri = urlparse(uri)
    try:
        s3.download_file(parsed_uri.netloc, parsed_uri.path[1:], path)
    except botocore.exceptions.ClientError:
        raise NotReadableError('Could not read {}'.format(uri))


def copy_to(src_path: str, dst_uri: str) -> None:
    s3 = get_session().client('s3')

    parsed_uri = urlparse(dst_uri)
    if os.path.isfile(src_path):
        try:
            s3.upload_file(src_path, parsed_uri.netloc,
                           parsed_uri.path[1:])
        except Exception as e:
            raise NotWritableError(
                'Could not write {}'.format(dst_uri)) from e
    else:
        sync_to_dir(src_path, dst_uri, delete=True)


def sync_from_dir(src_dir_uri: str,
                  dest_dir_uri: str,
                  delete: bool = False) -> None:
    command = ['aws', 's3', 'sync', src_dir_uri, dest_dir_uri]
    if delete:
        command.append('--delete')
    subprocess.run(command)


def sync_to_dir(src_dir_uri: str, dest_dir_uri: str,
                delete: bool = False) -> None:
    command = ['aws', 's3', 'sync', src_dir_uri, dest_dir_uri]
    if delete:
        command.append('--delete')
    subprocess.run(command)


# Code from https://alexwlchan.net/2017/07/listing-s3-keys/
def get_matching_s3_objects(bucket, prefix='', suffix=''):
    """
    Generate objects in an S3 bucket.

    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch objects whose key starts with
        this prefix (optional).
    :param suffix: Only fetch objects whose keys end with
        this suffix (optional).
    """
    import boto3
    s3 = boto3.client('s3')
    kwargs = {'Bucket': bucket}

    # If the prefix is a single string (not a tuple of strings), we can
    # do the filtering directly in the S3 API.
    if isinstance(prefix, str):
        kwargs['Prefix'] = prefix

    while True:

        # The S3 API response is a large blob of metadata.
        # 'Contents' contains information about the listed objects.
        resp = s3.list_objects_v2(**kwargs)

        try:
            contents = resp['Contents']
        except KeyError:
            return

        for obj in contents:
            key = obj['Key']
            if key.startswith(prefix) and key.endswith(suffix):
                yield obj

        # The S3 API is paginated, returning up to 1000 keys at a time.
        # Pass the continuation token into the next response, until we
        # reach the final page (when this field is missing).
        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break


def get_matching_s3_keys(bucket, prefix='', suffix=''):
    """
    Generate the keys in an S3 bucket.

    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch keys that start with this prefix (optional).
    :param suffix: Only fetch keys that end with this suffix (optional).
    """
    for obj in get_matching_s3_objects(bucket, prefix, suffix):
        yield obj['Key']


def list_paths(uri, ext=''):
    parsed_uri = urlparse(uri)
    bucket = parsed_uri.netloc
    prefix = os.path.join(parsed_uri.path[1:])
    keys = get_matching_s3_keys(bucket, prefix, suffix=ext)
    return [os.path.join('s3://', bucket, key) for key in keys]
