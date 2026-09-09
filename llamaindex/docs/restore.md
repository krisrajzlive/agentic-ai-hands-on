# Restore

You can restore an individual file or folder without a full machine restore.

Open the console, go to Protection then Recovery, pick the machine, choose a
recovery point by date, and click Browse files. Select the file and click
Restore. Choose Original location or Alternate location.

If you restore to the original location and the file still exists, Northwind
writes the restored copy next to it with a .restored suffix, so nothing is
overwritten.

File-level browse only works for recovery points created after the agent was
upgraded to version 4.2 or later. Older recovery points support full-image
restore only. Restoring to an alternate location needs free space of at least
1.5 times the restored data size.
