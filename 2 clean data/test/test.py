import mount_bucket
import bucket_util as bu

bu.setBucketLocation('~/bucket')
for file in bu.walk('data','temperature'):
    print file
