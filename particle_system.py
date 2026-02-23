"""
BlackRoad Particle System
GPU-friendly particle system simulation with ASCII rendering.
"""

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EmissionType(str, Enum):
    BURST = "burst"
    CONTINUOUS = "continuous"
    EXPLOSION = "explosion"


class FieldType(str, Enum):
    GRAVITY = "gravity"
    WIND = "wind"
    VORTEX = "vortex"
    ATTRACTOR = "attractor"
    REPULSOR = "repulsor"
    TURBULENCE = "turbulence"


# ---------------------------------------------------------------------------
# Particle
# ---------------------------------------------------------------------------

@dataclass
class Particle:
    """A single particle in the simulation."""
    pos_x: float
    pos_y: float
    vel_x: float
    vel_y: float
    life: float          # remaining life in seconds
    max_life: float      # total life at spawn
    size: float = 1.0
    color_r: float = 1.0
    color_g: float = 1.0
    color_b: float = 1.0
    color_a: float = 1.0
    mass: float = 1.0
    acc_x: float = 0.0
    acc_y: float = 0.0
    rotation: float = 0.0
    angular_velocity: float = 0.0
    active: bool = True
    _id: int = field(default_factory=lambda: random.randint(0, 2**31))

    @property
    def life_ratio(self) -> float:
        """0.0 (born) → 1.0 (dead)."""
        if self.max_life <= 0:
            return 1.0
        return 1.0 - (self.life / self.max_life)

    @property
    def alpha(self) -> float:
        """Fade out towards end of life."""
        return self.color_a * (self.life / self.max_life)

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "pos": (round(self.pos_x, 3), round(self.pos_y, 3)),
            "vel": (round(self.vel_x, 3), round(self.vel_y, 3)),
            "life": round(self.life, 4),
            "life_ratio": round(self.life_ratio, 3),
            "size": self.size,
            "color": (self.color_r, self.color_g, self.color_b, round(self.alpha, 3)),
            "mass": self.mass,
            "active": self.active,
        }


# ---------------------------------------------------------------------------
# Particle Config
# ---------------------------------------------------------------------------

@dataclass
class ParticleConfig:
    """Template for spawning particles."""
    min_life: float = 1.0
    max_life: float = 3.0
    min_vel_x: float = -1.0
    max_vel_x: float = 1.0
    min_vel_y: float = -1.0
    max_vel_y: float = 1.0
    min_size: float = 0.5
    max_size: float = 2.0
    min_mass: float = 0.8
    max_mass: float = 1.2
    color_r: float = 1.0
    color_g: float = 0.5
    color_b: float = 0.0
    color_a: float = 1.0
    spread_x: float = 5.0
    spread_y: float = 5.0
    min_angular_velocity: float = -0.5
    max_angular_velocity: float = 0.5

    def spawn(self, x: float, y: float) -> Particle:
        life = random.uniform(self.min_life, self.max_life)
        return Particle(
            pos_x=x + random.uniform(-self.spread_x, self.spread_x),
            pos_y=y + random.uniform(-self.spread_y, self.spread_y),
            vel_x=random.uniform(self.min_vel_x, self.max_vel_x),
            vel_y=random.uniform(self.min_vel_y, self.max_vel_y),
            life=life,
            max_life=life,
            size=random.uniform(self.min_size, self.max_size),
            color_r=self.color_r,
            color_g=self.color_g,
            color_b=self.color_b,
            color_a=self.color_a,
            mass=random.uniform(self.min_mass, self.max_mass),
            angular_velocity=random.uniform(self.min_angular_velocity, self.max_angular_velocity),
        )


# ---------------------------------------------------------------------------
# Force Fields
# ---------------------------------------------------------------------------

@dataclass
class ForceField:
    field_type: FieldType
    params: dict = field(default_factory=dict)
    enabled: bool = True

    def apply(self, p: Particle, dt: float):
        if not self.enabled:
            return
        if self.field_type == FieldType.GRAVITY:
            strength = self.params.get("strength", 9.8)
            p.acc_y -= strength
        elif self.field_type == FieldType.WIND:
            strength = self.params.get("strength", 2.0)
            direction = self.params.get("direction", 0.0)  # radians
            p.acc_x += strength * math.cos(direction)
            p.acc_y += strength * math.sin(direction)
        elif self.field_type == FieldType.VORTEX:
            cx = self.params.get("cx", 0.0)
            cy = self.params.get("cy", 0.0)
            strength = self.params.get("strength", 5.0)
            dx = p.pos_x - cx
            dy = p.pos_y - cy
            dist = math.sqrt(dx * dx + dy * dy) + 0.001
            p.acc_x += -dy * strength / (dist * dist)
            p.acc_y += dx * strength / (dist * dist)
        elif self.field_type == FieldType.ATTRACTOR:
            cx = self.params.get("cx", 0.0)
            cy = self.params.get("cy", 0.0)
            strength = self.params.get("strength", 10.0)
            dx = cx - p.pos_x
            dy = cy - p.pos_y
            dist = math.sqrt(dx * dx + dy * dy) + 0.001
            p.acc_x += strength * dx / (dist * dist)
            p.acc_y += strength * dy / (dist * dist)
        elif self.field_type == FieldType.REPULSOR:
            cx = self.params.get("cx", 0.0)
            cy = self.params.get("cy", 0.0)
            strength = self.params.get("strength", 10.0)
            dx = p.pos_x - cx
            dy = p.pos_y - cy
            dist = math.sqrt(dx * dx + dy * dy) + 0.001
            p.acc_x += strength * dx / (dist * dist)
            p.acc_y += strength * dy / (dist * dist)
        elif self.field_type == FieldType.TURBULENCE:
            strength = self.params.get("strength", 1.0)
            p.acc_x += random.uniform(-strength, strength)
            p.acc_y += random.uniform(-strength, strength)


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------

@dataclass
class Emitter:
    """Particle emitter."""
    x: float
    y: float
    rate_per_sec: float
    particle_config: ParticleConfig = field(default_factory=ParticleConfig)
    emission_type: EmissionType = EmissionType.CONTINUOUS
    active: bool = True
    _id: str = field(default_factory=lambda: f"emitter_{random.randint(1000, 9999)}")
    _accumulator: float = field(default=0.0, init=False, repr=False)
    _emitted_total: int = field(default=0, init=False, repr=False)

    def tick(self, dt: float) -> List[Particle]:
        """Advance timer and return newly spawned particles."""
        if not self.active:
            return []
        if self.emission_type == EmissionType.BURST:
            return []   # Bursts are triggered manually
        self._accumulator += dt * self.rate_per_sec
        spawned = []
        while self._accumulator >= 1.0:
            spawned.append(self.particle_config.spawn(self.x, self.y))
            self._accumulator -= 1.0
            self._emitted_total += 1
        return spawned

    def burst(self, count: int) -> List[Particle]:
        """Emit a burst of particles."""
        particles = [self.particle_config.spawn(self.x, self.y) for _ in range(count)]
        self._emitted_total += count
        return particles

    def explosion(self, count: int, speed: float = 20.0) -> List[Particle]:
        """Emit particles in all directions (explosion)."""
        particles = []
        for i in range(count):
            angle = 2 * math.pi * i / count + random.uniform(-0.3, 0.3)
            spd = speed * random.uniform(0.5, 1.5)
            p = self.particle_config.spawn(self.x, self.y)
            p.vel_x = math.cos(angle) * spd
            p.vel_y = math.sin(angle) * spd
            particles.append(p)
        self._emitted_total += count
        return particles

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "pos": (self.x, self.y),
            "type": self.emission_type.value,
            "rate": self.rate_per_sec,
            "emitted": self._emitted_total,
            "active": self.active,
        }


# ---------------------------------------------------------------------------
# Particle System
# ---------------------------------------------------------------------------

class ParticleSystem:
    """Main particle system manager."""

    def __init__(self, max_particles: int = 10000, bounds: Optional[Tuple[float, float, float, float]] = None):
        """
        bounds: (min_x, min_y, max_x, max_y) — particles outside are deactivated.
        """
        self.max_particles = max_particles
        self.bounds = bounds
        self.particles: List[Particle] = []
        self.emitters: Dict[str, Emitter] = {}
        self.fields: List[ForceField] = []
        self._frame: int = 0
        self._time: float = 0.0
        self._total_spawned: int = 0
        self._total_killed: int = 0

    # ------------------------------------------------------------------
    # Emitter management
    # ------------------------------------------------------------------

    def add_emitter(self, emitter: Emitter) -> str:
        self.emitters[emitter._id] = emitter
        return emitter._id

    def remove_emitter(self, emitter_id: str):
        self.emitters.pop(emitter_id, None)

    def set_emitter_position(self, emitter_id: str, x: float, y: float):
        if emitter_id in self.emitters:
            self.emitters[emitter_id].x = x
            self.emitters[emitter_id].y = y

    # ------------------------------------------------------------------
    # Force fields
    # ------------------------------------------------------------------

    def apply_force(self, force_x: float, force_y: float):
        """Apply a uniform force to all active particles."""
        for p in self.particles:
            if p.active:
                p.acc_x += force_x / p.mass
                p.acc_y += force_y / p.mass

    def apply_field(self, field_type: FieldType, params: Optional[dict] = None):
        """Add a persistent force field."""
        self.fields.append(ForceField(field_type=field_type, params=params or {}))

    def clear_fields(self):
        self.fields.clear()

    # ------------------------------------------------------------------
    # Spawn
    # ------------------------------------------------------------------

    def emit(self, emitter_id: str, count: int = 1):
        """Manually emit particles from an emitter."""
        emitter = self.emitters.get(emitter_id)
        if not emitter:
            return
        if emitter.emission_type == EmissionType.EXPLOSION:
            new_particles = emitter.explosion(count)
        else:
            new_particles = emitter.burst(count)
        self._add_particles(new_particles)

    def _add_particles(self, new_particles: List[Particle]):
        available = self.max_particles - len(self.particles)
        to_add = new_particles[:available]
        self.particles.extend(to_add)
        self._total_spawned += len(to_add)

    def spawn_at(self, x: float, y: float, count: int = 1, config: Optional[ParticleConfig] = None):
        """Spawn particles at an arbitrary position."""
        cfg = config or ParticleConfig()
        new_particles = [cfg.spawn(x, y) for _ in range(count)]
        self._add_particles(new_particles)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, dt: float):
        """Advance the simulation by dt seconds."""
        self._time += dt
        self._frame += 1

        # Tick emitters and collect new particles
        new_particles: List[Particle] = []
        for emitter in self.emitters.values():
            new_particles.extend(emitter.tick(dt))
        self._add_particles(new_particles)

        # Update existing particles
        alive: List[Particle] = []
        for p in self.particles:
            if not p.active:
                continue
            # Apply fields
            for field in self.fields:
                field.apply(p, dt)
            # Integrate velocity
            p.vel_x += p.acc_x * dt
            p.vel_y += p.acc_y * dt
            # Integrate position
            p.pos_x += p.vel_x * dt
            p.pos_y += p.vel_y * dt
            # Rotation
            p.rotation += p.angular_velocity * dt
            # Reset acceleration (forces re-applied each frame)
            p.acc_x = 0.0
            p.acc_y = 0.0
            # Age
            p.life -= dt
            if p.life <= 0:
                p.active = False
                self._total_killed += 1
                continue
            # Bounds check
            if self.bounds:
                min_x, min_y, max_x, max_y = self.bounds
                if not (min_x <= p.pos_x <= max_x and min_y <= p.pos_y <= max_y):
                    p.active = False
                    self._total_killed += 1
                    continue
            alive.append(p)

        self.particles = alive

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_frame(self, width: int = 80, height: int = 24) -> str:
        """Render current particle positions as ASCII art."""
        if not self.particles:
            return "\n".join(["." * width] * height)

        # Find world bounds for this frame
        xs = [p.pos_x for p in self.particles]
        ys = [p.pos_y for p in self.particles]
        world_min_x, world_max_x = min(xs), max(xs)
        world_min_y, world_max_y = min(ys), max(ys)
        world_w = world_max_x - world_min_x or 1.0
        world_h = world_max_y - world_min_y or 1.0

        GLYPHS = " ·:;+=xX$&@#"
        grid = [[" "] * width for _ in range(height)]

        for p in self.particles:
            col = int((p.pos_x - world_min_x) / world_w * (width - 1))
            row = int((p.pos_y - world_min_y) / world_h * (height - 1))
            col = max(0, min(width - 1, col))
            row = max(0, min(height - 1, row))
            alpha = p.alpha
            glyph_idx = int(alpha * (len(GLYPHS) - 1))
            glyph_idx = max(0, min(len(GLYPHS) - 1, glyph_idx))
            grid[row][col] = GLYPHS[glyph_idx]

        return "\n".join("".join(row) for row in grid)

    def export_frames(self, count: int, width: int = 40, height: int = 12, dt: float = 0.016) -> List[str]:
        """Simulate and export count frames as ASCII strings."""
        frames = []
        for _ in range(count):
            frames.append(self.render_frame(width, height))
            self.update(dt)
        return frames

    # ------------------------------------------------------------------
    # Stats & serialization
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        active = [p for p in self.particles if p.active]
        avg_life = sum(p.life for p in active) / len(active) if active else 0
        return {
            "frame": self._frame,
            "time": round(self._time, 4),
            "alive": len(active),
            "total_spawned": self._total_spawned,
            "total_killed": self._total_killed,
            "emitters": len(self.emitters),
            "fields": len(self.fields),
            "avg_remaining_life": round(avg_life, 4),
            "max_capacity": self.max_particles,
        }

    def snapshot(self) -> dict:
        return {
            "stats": self.get_stats(),
            "particles": [p.to_dict() for p in self.particles],
            "emitters": [e.to_dict() for e in self.emitters.values()],
        }

    def reset(self):
        self.particles.clear()
        self._frame = 0
        self._time = 0.0
        self._total_spawned = 0
        self._total_killed = 0


# ---------------------------------------------------------------------------
# Preset systems
# ---------------------------------------------------------------------------

def make_fire_system(x: float = 0, y: float = 0) -> ParticleSystem:
    """Create a fire particle system preset."""
    ps = ParticleSystem(max_particles=2000)
    cfg = ParticleConfig(
        min_life=0.5, max_life=1.5,
        min_vel_x=-2.0, max_vel_x=2.0,
        min_vel_y=5.0, max_vel_y=15.0,
        min_size=0.5, max_size=2.0,
        color_r=1.0, color_g=0.4, color_b=0.0,
        spread_x=2.0, spread_y=0.5,
    )
    emitter = Emitter(x=x, y=y, rate_per_sec=50, particle_config=cfg, emission_type=EmissionType.CONTINUOUS)
    ps.add_emitter(emitter)
    ps.apply_field(FieldType.TURBULENCE, {"strength": 0.5})
    return ps


def make_explosion_system(x: float = 0, y: float = 0, count: int = 200) -> ParticleSystem:
    """Create an explosion preset."""
    ps = ParticleSystem(max_particles=500)
    cfg = ParticleConfig(
        min_life=0.3, max_life=1.2,
        min_vel_x=-20, max_vel_x=20,
        min_vel_y=-20, max_vel_y=20,
        color_r=1.0, color_g=0.8, color_b=0.2,
        spread_x=0.5, spread_y=0.5,
    )
    emitter = Emitter(x=x, y=y, rate_per_sec=0, particle_config=cfg, emission_type=EmissionType.EXPLOSION)
    eid = ps.add_emitter(emitter)
    ps.emit(eid, count)
    ps.apply_field(FieldType.GRAVITY, {"strength": 9.8})
    return ps


def make_fountain_system(x: float = 0, y: float = 0) -> ParticleSystem:
    """Create a fountain preset."""
    ps = ParticleSystem(max_particles=1000)
    cfg = ParticleConfig(
        min_life=2.0, max_life=4.0,
        min_vel_x=-3.0, max_vel_x=3.0,
        min_vel_y=15.0, max_vel_y=25.0,
        color_r=0.3, color_g=0.6, color_b=1.0,
        spread_x=1.0, spread_y=0.2,
    )
    emitter = Emitter(x=x, y=y, rate_per_sec=30, particle_config=cfg)
    ps.add_emitter(emitter)
    ps.apply_field(FieldType.GRAVITY, {"strength": 9.8})
    return ps


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("=== BlackRoad Particle System Demo ===")

    # Basic system
    ps = ParticleSystem(max_particles=500)
    cfg = ParticleConfig(
        min_life=1.0, max_life=3.0,
        min_vel_x=-5.0, max_vel_x=5.0,
        min_vel_y=0.0, max_vel_y=10.0,
    )
    emitter = Emitter(x=0, y=0, rate_per_sec=20, particle_config=cfg)
    eid = ps.add_emitter(emitter)
    ps.apply_field(FieldType.GRAVITY, {"strength": 5.0})

    # Simulate
    for _ in range(60):
        ps.update(1 / 60)

    print(f"\nStats after 1s: {ps.get_stats()}")
    print("\nAscii frame (40x12):")
    print(ps.render_frame(40, 12))

    # Explosion
    print("\n[Explosion preset]")
    exp = make_explosion_system(0, 0, count=100)
    for _ in range(30):
        exp.update(1 / 30)
    print(f"Explosion stats: {exp.get_stats()}")
    print(exp.render_frame(60, 15))

    # Fire
    print("\n[Fire preset]")
    fire = make_fire_system()
    frames = fire.export_frames(5, width=40, height=10)
    for i, f in enumerate(frames):
        print(f"--- Frame {i} ---")
        print(f)

    # Manual spawn
    print("\n[Manual spawn]")
    ps2 = ParticleSystem()
    ps2.spawn_at(10, 10, count=50)
    ps2.apply_field(FieldType.VORTEX, {"cx": 10, "cy": 10, "strength": 3.0})
    ps2.update(0.5)
    print(ps2.get_stats())


if __name__ == "__main__":
    demo()
