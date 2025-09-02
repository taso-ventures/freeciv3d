#!/bin/bash

echo "=== Freeciv3D Docker Container Starting ==="

# Start all Freeciv-web services first (this starts MySQL, nginx, tomcat)
echo "Starting Freeciv-web services..."
/docker/scripts/start-freeciv-web.sh

# Then run database initialization
echo "Running database initialization..."
/docker/scripts/docker-init-db.sh || {
    echo "Database initialization failed, but continuing..."
    echo "You may need to run database initialization manually"
}

echo "=== Freeciv3D Container Ready ==="

exec "$@"
