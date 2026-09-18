# TH06-sized experiment geometry

The Pygame playfield is now 384 × 448. Its observation windows match the
native TH06 adapter: blue covers the full field at 8 × 8; yellow covers
204 × 204 at 16 × 16; red covers 64 × 64 at 64 × 64. The PCCM halo is 20 px.

The 26 existing level JSON files retain their original 600 × 700 coordinates.
`GameScene` scales them by 0.64 when loading, including enemy paths, bullet
speeds, wall sizes, collider radii and offsets, and bullet display sizes.
Sprite sheet crop dimensions and timing values are unchanged. To author a
level directly in the new coordinate system, put `"playfield_size": [384, 448]`
at the top level of its JSON.

Player movement, bullets, item motion, and collision sizes also scale by 0.64.
The red window changes from 128 to 64 px, which is a narrower field of view
than proportional scaling would give (about 82 px). Results from this setup
therefore need their own baseline runs; do not mix them with the 600 × 700
results. The model input tensor shapes stay the same. The Pygame window stays
1200 × 800 so its HUD and debug panels remain visible; the smaller playfield
primarily reduces the occupancy and PCCM spatial work.
