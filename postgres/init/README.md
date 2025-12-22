This folder is mounted into Postgres as `/docker-entrypoint-initdb.d`.

- Put `*.sql` scripts here if you want Postgres to initialize schemas/databases on first startup.
- Note: scripts only run the first time the `postgres_data` volume is created.
