@echo off
echo Creating UserFiles table...
%PSQL_PATH% -h %PGHOST% -p %PGPORT% -d %PGDATABASE% -U %PGUSER% -f create_user_files_table.sql
echo Done!
pause