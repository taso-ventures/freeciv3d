#!/bin/bash
# Docker database initialization script

set -e

echo "=== Starting Database Initialization ==="

# Configuration
DB_HOST="127.0.0.1"
DB_PORT="3306"
DB_USER="docker"
DB_PASSWORD="changeme"
DB_NAME="freeciv_web"

# Wait for MySQL to be ready
echo "Waiting for MySQL to be ready..."
for i in {1..60}; do
    if mysqladmin ping -h $DB_HOST -P $DB_PORT --silent 2>/dev/null; then
        echo "MySQL is ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "ERROR: MySQL failed to start within 60 seconds"
        exit 1
    fi
    echo "  Waiting for MySQL... ($i/60)"
    sleep 1
done

# Create database and user
echo "Setting up database and user..."
mysql -h $DB_HOST -P $DB_PORT -u root -e "
CREATE DATABASE IF NOT EXISTS $DB_NAME;
CREATE USER IF NOT EXISTS '$DB_USER'@'localhost' IDENTIFIED BY '$DB_PASSWORD';
CREATE USER IF NOT EXISTS '$DB_USER'@'%' IDENTIFIED BY '$DB_PASSWORD';
GRANT ALL PRIVILEGES ON $DB_NAME.* TO '$DB_USER'@'localhost';
GRANT ALL PRIVILEGES ON $DB_NAME.* TO '$DB_USER'@'%';
FLUSH PRIVILEGES;
" 2>/dev/null || {
    echo "Database setup completed or already exists"
}

# Test connection
echo "Testing database connection..."
mysql -h $DB_HOST -P $DB_PORT -u $DB_USER -p$DB_PASSWORD $DB_NAME -e "SELECT 1;" >/dev/null || {
    echo "ERROR: Cannot connect to database"
    exit 1
}

# Run migrations manually
FLYWAY_DIR="/docker/freeciv-web/src/main/resources/db/migration"
if [ -d "$FLYWAY_DIR" ]; then
    echo "Running database migrations..."
    for migration in $(ls $FLYWAY_DIR/V*.sql | sort); do
        echo "Running migration: $(basename $migration)"
        mysql -h $DB_HOST -P $DB_PORT -u $DB_USER -p$DB_PASSWORD $DB_NAME < "$migration" 2>/dev/null || {
            echo "Migration $(basename $migration) already applied or failed"
        }
    done
fi

# Register game servers
echo "Registering game servers..."
mysql -h $DB_HOST -P $DB_PORT -u $DB_USER -p$DB_PASSWORD $DB_NAME -e "
INSERT INTO servers (host, port, version, state, type, available, stamp) VALUES
('localhost', 6000, 'freeciv-web-devel', 'Pregame', 'singleplayer', 1, NOW()),
('localhost', 6001, 'freeciv-web-devel', 'Pregame', 'multiplayer', 1, NOW()),
('localhost', 6002, 'freeciv-web-devel', 'Pregame', 'singleplayer', 1, NOW()),
('localhost', 6003, 'freeciv-web-devel', 'Pregame', 'singleplayer', 1, NOW()),
('localhost', 6004, 'freeciv-web-devel', 'Pregame', 'multiplayer', 1, NOW())
ON DUPLICATE KEY UPDATE available=1, stamp=NOW();
" 2>/dev/null

echo "=== Database Initialization Complete ==="