-- Initialize game servers for Docker environment
-- This migration ensures that game servers are available in the metaserver database
-- for local development environments

-- Register initial game servers for ports 6000-6009
INSERT IGNORE INTO servers (host, port, version, state, type, available, stamp) VALUES
('localhost', 6000, 'freeciv-web-devel', 'Pregame', 'singleplayer', 1, NOW()),
('localhost', 6001, 'freeciv-web-devel', 'Pregame', 'multiplayer', 1, NOW()),
('localhost', 6002, 'freeciv-web-devel', 'Pregame', 'singleplayer', 1, NOW()),
('localhost', 6003, 'freeciv-web-devel', 'Pregame', 'singleplayer', 1, NOW()),
('localhost', 6004, 'freeciv-web-devel', 'Pregame', 'multiplayer', 1, NOW()),
('localhost', 6005, 'freeciv-web-devel', 'Pregame', 'singleplayer', 1, NOW()),
('localhost', 6006, 'freeciv-web-devel', 'Pregame', 'singleplayer', 1, NOW()),
('localhost', 6007, 'freeciv-web-devel', 'Pregame', 'multiplayer', 1, NOW()),
('localhost', 6008, 'freeciv-web-devel', 'Pregame', 'singleplayer', 1, NOW()),
('localhost', 6009, 'freeciv-web-devel', 'Pregame', 'singleplayer', 1, NOW());

-- Note: This is for development environments only
-- In production, servers should register themselves dynamically