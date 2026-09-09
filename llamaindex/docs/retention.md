# Retention

Northwind Cloud Backup keeps recovery points on a retention schedule.

Under the default Standard schedule, daily recovery points are kept for 30 days,
weekly recovery points for 12 weeks, and monthly recovery points for 12 months.

Administrators can create a custom retention policy. The minimum retention for any
tier is 7 days. The maximum is 7 years, which is the limit of the underlying
immutable storage.

Pruning runs once per day at 02:00 in the account's time zone. A recovery point
is only removed when every file version it uniquely holds has aged out of all
retention tiers. Legal hold overrides retention entirely.
