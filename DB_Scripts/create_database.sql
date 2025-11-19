------------------------------------------
-- create_database.sql
-- Script to create AKAvR_DB database
------------------------------------------

-- Create database if it doesn't exist
CREATE DATABASE "AKAvR_DB";

-- Display success message
\echo 'Database AKAvR_DB created successfully!'

-- Show all databases to verify
SELECT datname as "Database Name" 
FROM pg_database 
WHERE datname IN ('postgres', 'AKAvR_DB')
ORDER BY datname;