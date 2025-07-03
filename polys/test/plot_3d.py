#!/usr/bin/env python

# test/cli2json.py test123 -x unit_circle,coeff7 -p poly_362 -z rev -s solve  | test/json2plot.py 10000
import numpy as np
import pygfx as gfx
from wgpu.gui.offscreen import WgpuCanvas
import imageio 

def norm(x):
     max_x = np.max(x)
     min_x = np.min(x)
     if max_x-min_x>0:
        return ((x-min_x)/(max_x-min_x)).astype(np.float32)
     else:
         return x.astype(np.float32)
     
def hsv_to_rgb_numpy(hsv):
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    i = (h * 6).astype(int)
    f = (h * 6) - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i = i % 6
    conditions = [i == k for k in range(6)]
    rgb = np.zeros(hsv.shape)
    rgb[conditions[0]] = np.stack([v, t, p], axis=-1)[conditions[0]]
    rgb[conditions[1]] = np.stack([q, v, p], axis=-1)[conditions[1]]
    rgb[conditions[2]] = np.stack([p, v, t], axis=-1)[conditions[2]]
    rgb[conditions[3]] = np.stack([p, q, v], axis=-1)[conditions[3]]
    rgb[conditions[4]] = np.stack([t, p, v], axis=-1)[conditions[4]]
    rgb[conditions[5]] = np.stack([v, p, q], axis=-1)[conditions[5]]
    return rgb


res_z = np.load('myresult.npz')
results = res_z['arr_0'].astype(np.float32)
params = results[:, [0,1]].astype(np.float32)
height = np.max(params[:,0])+1
width = np.max(params[:,1])+1
print(f"raster:{height},{width}")
params[:,0] = params[:,0]/np.max(params[:,0])-0.5
params[:,1] = params[:,1]/np.max(params[:,1])-0.5
points = results[:, [0, 2, 3]].astype(np.float32)
points[:,0] = points[:,0]/(height-1)/4
points[:,1] = points[:,1]/(height-1)-0.5
points[:,2] = points[:,2]/(height-1)-0.5
print(f"points 0:{np.min(points[:,0])},{np.max(points[:,0])}")
print(f"points 1:{np.min(points[:,1])},{np.max(points[:,1])}")
print(f"points 2:{np.min(points[:,2])},{np.max(points[:,2])}")
hsv = np.zeros((points.shape[0],3),dtype=np.float32)
hue = (np.angle(params[:,0]+1j*params[:,1])/np.pi+1.0)/2.0
hsv[:,0]= (np.sin(2*np.pi*hue)+1.0) * 0.5 + 0
hsv[:,1]=((50 * np.abs(params[:,0]+1j*params[:,1])) % 1) * 0.75 + 0.24
hsv[:,2]=1.0
rgb=hsv_to_rgb_numpy(hsv).astype(np.float32)
print(f"r: {np.min(rgb[:,0])}-{np.max(rgb[:,0])}")
print(f"g: {np.min(rgb[:,1])}-{np.max(rgb[:,1])}")
print(f"b: {np.min(rgb[:,2])}-{np.max(rgb[:,2])}")

alpha = np.ones((rgb.shape[0],1), dtype=np.float32)*0.01
rgba = np.concatenate([rgb, alpha], axis=1)
print(f"rgba:{rgba.shape}")
print(f"h: {np.min(hsv[:,0])}-{np.max(hsv[:,0])}")
print(f"rgb: {rgb.shape}")

centroid = points.mean(axis=0)
mins = points.min(axis=0)
maxs = points.max(axis=0)
size = np.linalg.norm(maxs - mins)
distance = size * 0.5  # tweak this multiplier

geometry = gfx.Geometry(positions=points,colors=rgba)
material = gfx.PointsMaterial(size=1,color_mode="vertex")
points_obj = gfx.Points(geometry, material)

scene = gfx.Scene()
scene.add(points_obj)

camera = gfx.OrthographicCamera(1, 1)
canvas = WgpuCanvas(size=(1000,1000),pixel_ratio=1)
renderer = gfx.renderers.WgpuRenderer(canvas)

print("logical :", canvas.get_logical_size())   # (1000, 1000)
print("ratio   :", canvas.get_pixel_ratio())    # 1  (should not be 2)
print("physical:", canvas.get_physical_size())  # (1000, 1000)
x = centroid[0] + distance 
y = centroid[1] + distance 
z = centroid[2] + distance 
camera.local.position = (x, y, z)
camera.look_at(tuple(centroid))
renderer.render(scene, camera)
img = renderer.snapshot()
print(img.shape)                                # (1000, 1000, 4)  <-- if not, wrong canvas

frames = []
n_frames = 1000
colors = geometry.colors.data
positions = geometry.positions.data
location = norm(positions[:,0])
for i in range(n_frames):
    if i % (n_frames//100) ==0:
        print(f"fames: {round(100*i/n_frames,2)}%")
    a = i/(n_frames-1)
    angle = 2 * np.pi * a
    x = centroid[0] + distance #* np.cos(angle) 
    y = centroid[1] + distance * 0 #* np.sin(2*angle) 
    z = centroid[2] + distance * 0 #* np.sin(angle)
    camera.local.position = (x, y, z)
    camera.look_at(tuple(centroid))
    colors[:,3] = np.exp(-0.5 *((location-a)/0.01)**2)
    geometry.colors.update_range()
    renderer.render(scene, camera)
    img = renderer.snapshot()
    frames.append(img)


with imageio.get_writer(
    "rotation.mp4", 
    fps=50,
    codec='libx264',
    quality=None,                    
    output_params=[
        "-crf", "18",
        "-preset", "slow",
        "-pix_fmt","yuv444p"
    ]
) as writer:
    for img in frames:
        writer.append_data(img)


    
