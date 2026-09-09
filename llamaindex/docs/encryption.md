# Encryption

Backup data is encrypted with AES-256 before it leaves the protected machine.
Traffic between the agent and the service uses TLS 1.2 or higher.

There are two key modes, chosen when a machine is first protected:

- Service-managed keys (default): Northwind holds the key in an HSM. Recovery
  works from any device after sign-in.
- Customer-managed passphrase: the passphrase never leaves your machine and
  Northwind stores only a salted verifier. If you lose this passphrase the data
  is unrecoverable. Support cannot reset it.

Key mode cannot be changed after the first backup. To switch, remove protection
and re-enroll the machine, which starts a new full backup.
