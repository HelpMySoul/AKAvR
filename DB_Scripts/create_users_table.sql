------------------------------------------
-- create_users_table.sql
------------------------------------------

-- Users table creation for PostgreSQL

CREATE TABLE IF NOT EXISTS "Users" (
    "Id" SERIAL PRIMARY KEY,
    "Email" VARCHAR(255) NOT NULL UNIQUE,
    "Username" VARCHAR(100) NOT NULL UNIQUE,
    "Password" VARCHAR(255) NOT NULL,
    "CreatedAt" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    "UpdatedAt" TIMESTAMP WITH TIME ZONE NULL,
    "IsActive" BOOLEAN DEFAULT TRUE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS "IX_Users_Email" ON "Users"("Email");
CREATE INDEX IF NOT EXISTS "IX_Users_Username" ON "Users"("Username");
CREATE INDEX IF NOT EXISTS "IX_Users_CreatedAt" ON "Users"("CreatedAt");
CREATE INDEX IF NOT EXISTS "IX_Users_IsActive" ON "Users"("IsActive");

-- Table and column comments
COMMENT ON TABLE "Users" IS 'Table for storing user data';
COMMENT ON COLUMN "Users"."Id" IS 'Unique user identifier';
COMMENT ON COLUMN "Users"."Email" IS 'User email (unique)';
COMMENT ON COLUMN "Users"."Username" IS 'Username (unique)';
COMMENT ON COLUMN "Users"."Password" IS 'User password hash';
COMMENT ON COLUMN "Users"."CreatedAt" IS 'Record creation timestamp';
COMMENT ON COLUMN "Users"."UpdatedAt" IS 'Last update timestamp';
COMMENT ON COLUMN "Users"."IsActive" IS 'User active flag';

-- Verify table structure
SELECT 
    table_name,
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns 
WHERE table_name = 'Users' 
ORDER BY ordinal_position;