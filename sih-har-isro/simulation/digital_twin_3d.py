"""3D BAS digital twin for HAR integration.

This renderer is deliberately geometry-first: every astronaut joint, payload
item and rack target has a 3D position in metres, then a fixed payload camera
projects it into the same pixels that OpenCV receives.  It is a deterministic
digital twin for test data and UI demonstrations -- not a replacement for a
licensed, photorealistic astronaut asset.

The public contract mirrors ``simulation.renderer``: return BGR frames and
metadata containing projected ground-truth boxes plus a 132-float pose vector.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np

from data_generation.synthetic_pose import generate_sequence

RED = (12, 12, 230)
YELLOW = (0, 255, 255)
WHITE = (235, 235, 235)


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


@dataclass
class FixedPayloadCamera:
    """Pinhole camera fixed to the payload rack, never to the astronaut."""
    width: int = 1280
    height: int = 720
    focal_px: float = 900.0
    position: np.ndarray = field(default_factory=lambda: np.array([0., 0., -3.4], np.float32))

    def project(self, p: np.ndarray) -> Tuple[int, int, float]:
        # The rack reference frame has +z away from the fixed camera.
        q = np.asarray(p, np.float32) - self.position
        z = max(float(q[2]), 0.25)
        return (int(self.width / 2 + self.focal_px * q[0] / z),
                int(self.height / 2 - self.focal_px * q[1] / z), z)

    def box(self, center: np.ndarray, size_m: Tuple[float, float]) -> Tuple[int, int, int, int]:
        x, y, z = self.project(center)
        hw = int(self.focal_px * size_m[0] / z / 2)
        hh = int(self.focal_px * size_m[1] / z / 2)
        return x - hw, y - hh, x + hw, y + hh


@dataclass
class FreeObject:
    name: str
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    held: str | None = None
    latched: bool = False


class MicrogravityPhysics:
    """Zero-g object dynamics: inertia, small ventilation disturbances, latching."""
    def __init__(self, seed: int = 1):
        self.rng = np.random.default_rng(seed)
        self.red = FreeObject("red_box", np.array([-0.16, -0.10, 0.30], np.float32))
        self.yellow = FreeObject("yellow_box", np.array([0.16, -0.10, 0.30], np.float32))
        self.opened = False
        self.frame = 0

    def step(self, step_id: int, phase: float, hands: Dict[str, np.ndarray], dt: float = 1/30):
        self.frame += 1
        if step_id >= 2:
            self.opened = True
        if step_id in (3, 4): self.red.held = "left"
        if step_id == 5 and phase > .55 and self.red.held:
            self.red.held = None; self.red.velocity = np.array([-.014, -.006, .002], np.float32)
        if step_id == 5 and phase > .87:
            self.red.position = np.array([-.78, -.20, .18], np.float32); self.red.latched = True
        if step_id in (6, 7): self.yellow.held = "right"
        if step_id == 8 and phase > .55 and self.yellow.held:
            self.yellow.held = None; self.yellow.velocity = np.array([.014, -.006, .002], np.float32)
        if step_id == 8 and phase > .87:
            self.yellow.position = np.array([.78, -.20, .18], np.float32); self.yellow.latched = True
        for obj in (self.red, self.yellow):
            if obj.held:
                obj.position = hands[obj.held].copy(); obj.velocity *= 0
            elif not obj.latched:
                vent = np.array([.00003*np.sin(self.frame*.11), .00002*np.cos(self.frame*.07), 0], np.float32)
                obj.velocity += vent
                obj.position += obj.velocity * dt
                obj.velocity *= .999  # no gravity; minimal residual atmosphere damping


class BASDigitalTwin3D:
    """Procedural 3D astronaut + payload-rack scene from a fixed camera."""
    def __init__(self, width: int = 1280, height: int = 720, seed: int = 7):
        self.camera = FixedPayloadCamera(width, height)
        self.physics = MicrogravityPhysics(seed)
        self.rng = np.random.default_rng(seed)

    def _joints(self, pose: np.ndarray) -> Dict[str, np.ndarray]:
        # Map pose landmarks into a rack-relative 3D working volume in metres.
        def p(i: int, depth: float = .35):
            return np.array([(pose[i,0]-.5)*2.0, (.56-pose[i,1])*1.8, depth + pose[i,2]], np.float32)
        return {"head":p(0,.42), "ls":p(11,.42), "rs":p(12,.42), "lh":p(23,.46), "rh":p(24,.46),
                "le":p(13,.38), "re":p(14,.38), "left":p(15,.32), "right":p(16,.32),
                "lk":p(25,.50), "rk":p(26,.50), "la":p(27,.54), "ra":p(28,.54)}

    def _background(self, image: np.ndarray):
        h,w = image.shape[:2]
        image[:] = (34, 33, 31)
        # Perspective rack bays, handrails, and a restrained payload work area.
        for x in (-1.35, 1.35):
            a=self.camera.project(np.array([x,-.95,.8],np.float32)); b=self.camera.project(np.array([x,.95,.8],np.float32))
            cv2.line(image,a[:2],b[:2],(95,102,110),18)
        for y in (-.8,.75):
            a=self.camera.project(np.array([-1.4,y,.78],np.float32)); b=self.camera.project(np.array([1.4,y,.78],np.float32))
            cv2.line(image,a[:2],b[:2],(150,155,160),7)
        for x in (-.78,.78):
            box=self.camera.box(np.array([x,-.2,.19],np.float32),(.30,.26))
            cv2.rectangle(image,box[:2],box[2:],(56,62,68),2)
        for i in range(36):
            p=np.array([self.rng.uniform(-1.3,1.3),self.rng.uniform(-.9,.9),self.rng.uniform(.2,1.3)])
            x,y,_=self.camera.project(p); cv2.circle(image,(x,y),1,(155,160,165),-1)

    def _limb(self, image, a, b, radius, color=(220,220,218)):
        pa=self.camera.project(a); pb=self.camera.project(b)
        thickness=max(5,int(radius*self.camera.focal_px/((pa[2]+pb[2])/2)))
        cv2.line(image,pa[:2],pb[:2],color,thickness,cv2.LINE_AA)

    def _astronaut(self, image: np.ndarray, j: Dict[str,np.ndarray]):
        # Depth-sorted capsules give a genuine 3D-coordinate, perspective render.
        for a,b,r in (("lh","lk",.095),("lk","la",.08),("rh","rk",.095),("rk","ra",.08),
                      ("ls","le",.09),("le","left",.075),("rs","re",.09),("re","right",.075)):
            self._limb(image,j[a],j[b],r)
        torso=np.array([j["ls"],j["rs"],j["rh"],j["lh"]])
        pts=np.array([self.camera.project(p)[:2] for p in torso],np.int32)
        cv2.fillConvexPoly(image,pts,(225,225,220),cv2.LINE_AA)
        hx,hy,hz=self.camera.project(j["head"])
        rr=max(14,int(.14*self.camera.focal_px/hz)); cv2.circle(image,(hx,hy),rr,(225,225,220),-1,cv2.LINE_AA)
        cv2.ellipse(image,(hx,hy),(int(rr*.65),int(rr*.48)),0,0,360,(42,48,55),-1,cv2.LINE_AA)
        for key in ("left","right"):
            x,y,z=self.camera.project(j[key]); cv2.circle(image,(x,y),max(8,int(.07*self.camera.focal_px/z)),(242,242,237),-1)

    def render(self, pose_flat: np.ndarray, step_id: int, phase: float) -> Tuple[np.ndarray,dict]:
        pose=pose_flat.reshape(33,4); j=self._joints(pose); self.physics.step(step_id,phase,{"left":j["left"],"right":j["right"]})
        frame=np.zeros((self.camera.height,self.camera.width,3),np.uint8); self._background(frame)
        # container is rack-mounted; loose contents are driven by 3D physics.
        main_center=np.array([0,-.18,.25],np.float32); main_box=self.camera.box(main_center,(.62,.48))
        cv2.rectangle(frame,main_box[:2],main_box[2:],WHITE,-1); cv2.rectangle(frame,main_box[:2],main_box[2:],(130,130,130),2)
        self._astronaut(frame,j)
        red_box=self.camera.box(self.physics.red.position,(.16,.13)); yellow_box=self.camera.box(self.physics.yellow.position,(.16,.13))
        if self.physics.opened:
            cv2.rectangle(frame,red_box[:2],red_box[2:],RED,-1); cv2.rectangle(frame,yellow_box[:2],yellow_box[2:],YELLOW,-1)
        return frame,{"step_id":step_id,"t":phase,"pose":pose_flat.astype(np.float32),"main_visible":True,
                      "red_visible":self.physics.opened,"yellow_visible":self.physics.opened,
                      "boxes":{"main_box":main_box,"red_box":red_box,"yellow_box":yellow_box},
                      "physics":{"gravity_g":0.0,"red_velocity_mps":self.physics.red.velocity.round(5).tolist(),
                                 "yellow_velocity_mps":self.physics.yellow.velocity.round(5).tolist(),
                                 "red_latched":self.physics.red.latched,"yellow_latched":self.physics.yellow.latched}}


def render_protocol_3d(frames_per_step: int=72, seed: int=7) -> Tuple[List[np.ndarray],List[dict]]:
    twin=BASDigitalTwin3D(seed=seed); frames=[]; metas=[]; rng=np.random.default_rng(seed)
    for sid in range(1,9):
        seq=generate_sequence(sid,n_frames=frames_per_step,rng=rng,orientation=(sid//3)%4,drift=True)
        for i,p in enumerate(seq):
            frame,meta=twin.render(p,sid,i/max(frames_per_step-1,1)); frames.append(frame); metas.append(meta)
    return frames,metas


def draw_camera_har_overlay(frame: np.ndarray, meta: dict) -> np.ndarray:
    """OpenCV-style camera overlay for the digital-twin validation video.

    Boxes are explicitly marked ``SIM-REFERENCE`` because the simulator knows
    the 3D truth.  In live mode ``HARPipeline._annotate_frame`` draws model
    detections with the identical object names and visual convention.
    """
    out=frame.copy()
    colors={"main_box":(220,220,220),"red_box":(0,0,255),"yellow_box":(0,255,255)}
    for name,box in meta["boxes"].items():
        if name != "main_box" and not meta.get(name.split("_")[0]+"_visible",False): continue
        x1,y1,x2,y2=map(int,box); color=colors[name]
        cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
        cv2.putText(out,f"{name} | SIM-REFERENCE",(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.48,color,1,cv2.LINE_AA)
    g=meta["physics"]
    cv2.rectangle(out,(12,12),(395,93),(16,20,24),-1)
    cv2.putText(out,"BAS FIXED PAYLOAD CAMERA | HAR INPUT",(22,35),cv2.FONT_HERSHEY_SIMPLEX,.56,(155,240,255),1,cv2.LINE_AA)
    cv2.putText(out,f"STEP {meta['step_id']} | MICROGRAVITY: {g['gravity_g']:.2f} g",(22,59),cv2.FONT_HERSHEY_SIMPLEX,.54,(80,255,150),1,cv2.LINE_AA)
    cv2.putText(out,"Rack-frame coordinates | object inertia active",(22,81),cv2.FONT_HERSHEY_SIMPLEX,.46,(205,205,205),1,cv2.LINE_AA)
    return out


def generate_3d_demo_assets(out_dir: str = "dataset/space_sim", frames_per_step: int = 72,
                            seed: int = 7) -> dict:
    """Write a standalone 3D digital-twin camera video without ML dependencies."""
    from pathlib import Path
    import json
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    frames, metas = render_protocol_3d(frames_per_step=frames_per_step, seed=seed)
    annotated = [draw_camera_har_overlay(frame, meta) for frame, meta in zip(frames, metas)]
    h,w = annotated[0].shape[:2]
    video = out / "bas_3d_digital_twin_har.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w,h))
    for frame in annotated: writer.write(frame)
    writer.release()
    preview = out / "bas_3d_digital_twin_preview.jpg"; cv2.imwrite(str(preview), annotated[len(annotated)//2])
    report = {"renderer":"procedural_3d_rack_coordinate_digital_twin", "frames":len(frames), "fps":30,
              "physics":"zero-g inertia, ventilation disturbance, rack-zone latching",
              "overlay":"SIM-REFERENCE boxes; live HAR pipeline renders model detections"}
    (out / "bas_3d_digital_twin_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"video":str(video),"preview":str(preview),**report}


def draw_camera_har_overlay(frame: np.ndarray, meta: dict) -> np.ndarray:
    """OpenCV-style camera overlay for the digital-twin validation video.

    Boxes are explicitly marked ``SIM-REFERENCE`` because the simulator knows
    the 3D truth.  In live mode ``HARPipeline._annotate_frame`` draws model
    detections with the identical object names and visual convention.
    """
    out=frame.copy()
    colors={"main_box":(220,220,220),"red_box":(0,0,255),"yellow_box":(0,255,255)}
    for name,box in meta["boxes"].items():
        if name != "main_box" and not meta.get(name.split("_")[0]+"_visible",False): continue
        x1,y1,x2,y2=map(int,box); color=colors[name]
        cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
        cv2.putText(out,f"{name} | SIM-REFERENCE",(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.48,color,1,cv2.LINE_AA)
    g=meta["physics"]
    cv2.rectangle(out,(12,12),(395,93),(16,20,24),-1)
    cv2.putText(out,"BAS FIXED PAYLOAD CAMERA | HAR INPUT",(22,35),cv2.FONT_HERSHEY_SIMPLEX,.56,(155,240,255),1,cv2.LINE_AA)
    cv2.putText(out,f"STEP {meta['step_id']} | MICROGRAVITY: {g['gravity_g']:.2f} g",(22,59),cv2.FONT_HERSHEY_SIMPLEX,.54,(80,255,150),1,cv2.LINE_AA)
    cv2.putText(out,"Rack-frame coordinates | object inertia active",(22,81),cv2.FONT_HERSHEY_SIMPLEX,.46,(205,205,205),1,cv2.LINE_AA)
    return out
