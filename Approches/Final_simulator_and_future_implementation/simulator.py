import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import segmentation_models_pytorch as smp
import yt_dlp  # only needed if streaming from YouTube

# --------------------------
# Device selection (CUDA if available, otherwise CPU)
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# YOLOv5 detection setup
# Loads a YOLOv5x model for object detection (used for finding cars/signs)
# --------------------------
yolo_weights = r"D:\night_segmentation_new\yolov5\runs\train\exp6\weights\best.pt"
yolo_repo    = r"D:\night_segmentation_new\yolov5"
yolo_model = torch.hub.load(
    'ultralytics/yolov5',
    'yolov5x',
    pretrained=True
).to(device).eval()

# --------------------------
# UNet segmentation setup
# Preprocessing stats for normalization
# --------------------------
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

# CLAHE object for local histogram equalization
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# -------------------------------------------------------
# Weather-specific preprocessing functions
# -------------------------------------------------------
def fog_preprocess(img_bgr, t0=0.1, omega=0.95, r=3):
    """
    Simple, fast dark-channel dehaze for foggy scenes.
    Args: img_bgr: input BGR image, returns dehazed BGR image.
    """
    I = img_bgr.astype(np.float32) / 255.0
    # Estimate air-light (atmospheric light)
    flat = I.reshape(-1, 3)
    k = max(1, int(0.001 * flat.shape[0]))
    idx = np.argpartition(flat.sum(1), -k)[-k:]
    A = flat[idx].mean(0)
    # Dark channel
    dc = cv2.erode(I.min(2), np.ones((r, r)))
    t  = 1.0 - omega * dc
    t  = np.clip(t, t0, 1)
    # Recover scene radiance
    J = (I - A) / t[..., None] + A
    return np.uint8(np.clip(J * 255, 0, 255))

def rain_preprocess(img_bgr, v_clip=0.85, gamma=0.8):
    """Suppress specular highlights (e.g. from rain) in a BGR frame."""
    hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    spec   = (s < 40) & (v > v_clip*255)
    v      = np.where(spec, v*gamma, v)
    hsv    = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def snow_preprocess(img_bgr, gamma=1.4):
    """Darken and equalize bright snowy regions in a BGR frame."""
    img = np.power(img_bgr / 255.0, gamma)
    img = (img * 255).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _CLAHE.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# --------------------------
# Load UNet segmentation model from checkpoint
# --------------------------
def load_unet_model(model_path, num_classes=3, device=torch.device("cpu")):
    net = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=num_classes
    ).to(device)
    state = torch.load(model_path, map_location=device)
    net.load_state_dict(state)
    net.eval()
    return net

# --------------------------
# Apply UNet segmentation model on a frame, using pre/post-processing for weather
# --------------------------
def process_frame_unet(frame, net, device, target_size=(1392,512), snow_prob=0.0, rain_prob=0.0, fog_prob=0.0):
    # Apply weather-specific pre-processing if above thresholds
    if snow_prob > 0.55:
        frame = snow_preprocess(frame)
    if rain_prob > 0.55:
        frame = rain_preprocess(frame)
    if fog_prob  > 0.55:
        frame = fog_preprocess(frame)
    # Prepare input for model
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil    = Image.fromarray(rgb).resize(target_size, Image.BILINEAR)
    tensor = TF.to_tensor(pil)
    tensor = TF.normalize(tensor, IMG_MEAN, IMG_STD).unsqueeze(0).to(device)
    with torch.no_grad():
        out = net(tensor)
    return out.argmax(dim=1).squeeze(0).cpu().numpy()

# --------------------------
# Segmentation mask coloring (background=black, road=red, lane=green)
# --------------------------
def colorize_mask(mask):
    cmap = {0:(0,0,0), 1:(0,0,255), 2:(0,255,0)}
    h,w  = mask.shape
    cm   = np.zeros((h,w,3), dtype=np.uint8)
    for cls,col in cmap.items():
        cm[mask==cls] = col
    return cm

# --------------------------
# Rain classifier using ResNet-18
# --------------------------
rain_model = models.resnet18(pretrained=False)
rain_model.fc = nn.Linear(rain_model.fc.in_features, 2)
rain_state = torch.load(
    r"D:\praca_magisterska\Adverse_weather_detection\model_rain.pth",
    map_location=device
)
rain_model.load_state_dict(rain_state)
rain_model = rain_model.to(device).eval()

def detect_rain(frame, model, device, size=(224,224)):
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil    = Image.fromarray(rgb).resize(size, Image.BILINEAR)
    tensor = TF.to_tensor(pil)
    tensor = TF.normalize(tensor, IMG_MEAN, IMG_STD).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return {'no_rain': float(probs[0]), 'rain': float(probs[1])}

# --------------------------
# Fog classifier using ResNet-18
# --------------------------
fog_model = models.resnet18(pretrained=False)
fog_model.fc = nn.Linear(fog_model.fc.in_features, 2)
fog_state = torch.load(
    r"D:\praca_magisterska\Adverse_weather_detection\model_fog.pth",
    map_location=device
)
fog_model.load_state_dict(fog_state)
fog_model = fog_model.to(device).eval()

def detect_fog(frame, model, device, size=(224,224)):
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil    = Image.fromarray(rgb).resize(size, Image.BILINEAR)
    tensor = TF.to_tensor(pil)
    tensor = TF.normalize(tensor, IMG_MEAN, IMG_STD).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return {'no_fog': float(probs[0]), 'fog': float(probs[1])}

# --------------------------
# Night classifier using ResNet-18
# --------------------------
night_model = models.resnet18(pretrained=False)
night_model.fc = nn.Linear(night_model.fc.in_features, 2)
night_state = torch.load(
    r"D:\praca_magisterska\Adverse_weather_detection\model_night.pth",
    map_location=device
)
night_model.load_state_dict(night_state)
night_model = night_model.to(device).eval()

def detect_night(frame, model, device, size=(224,224), gamma=1.8):
    dark = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)/255.0
    dark = np.power(dark, gamma)
    dark = (dark*255).astype(np.uint8)
    pil  = Image.fromarray(dark).resize(size, Image.BILINEAR)
    tensor = TF.to_tensor(pil)
    tensor = TF.normalize(tensor, IMG_MEAN, IMG_STD).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return {'day': float(probs[0]), 'night': float(probs[1])}

# --------------------------
# Snow classifier using ResNet-18
# --------------------------
snow_model = models.resnet18(pretrained=False)
snow_model.fc = nn.Linear(snow_model.fc.in_features, 2)
snow_state = torch.load(
    r"D:\praca_magisterska\Adverse_weather_detection\model_snow.pth",
    map_location=device
)
snow_model.load_state_dict(snow_state)
snow_model = snow_model.to(device).eval()

def detect_snow(frame, model, device, size=(224,224)):
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil    = Image.fromarray(rgb).resize(size, Image.BILINEAR)
    tensor = TF.to_tensor(pil)
    tensor = TF.normalize(tensor, IMG_MEAN, IMG_STD).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return {'no_snow': float(probs[0]), 'snow': float(probs[1])}

# --------------------------
# Bird’s‐eye view transformation (IPM)
# --------------------------
def get_bird_eye_view(img, src_pts, dst_pts, size):
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, size, flags=cv2.INTER_NEAREST)

# --------------------------
# Load segmentation model
# --------------------------
unet_model = load_unet_model(
    r"D:\night_segmentation_new\unet34_night_final_resized.pth",
    num_classes=3, device=device
)

# --------------------------
# Geometry and light beam helper functions (for matrix headlights simulation)
# --------------------------
def lerp(p1, p2, t):
    return ((1-t)*p1[0]+t*p2[0], (1-t)*p1[1]+t*p2[1])

def bilerp(quad, s, t):
    top    = lerp(quad[0], quad[1], s)
    bottom = lerp(quad[3], quad[2], s)
    return lerp(top, bottom, t)

def compute_grid_vertices(quad, cols, rows):
    verts = []
    for i in range(rows+1):
        t = i/rows
        row=[]
        for j in range(cols+1):
            s=j/cols
            x,y=bilerp(quad,s,t)
            row.append((int(x),int(y)))
        verts.append(row)
    return verts

def fill_gradient_polygon(img, poly, nc, fc, color, alpha):
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask,[poly],255)
    x,y,wb,hb=cv2.boundingRect(poly)
    X=np.arange(x,x+wb); Y=np.arange(y,y+hb)
    XX,YY=np.meshgrid(X,Y)
    inside=mask[y:y+hb,x:x+wb]>0
    beam_vec=np.array(fc,np.float32)-np.array(nc,np.float32)
    L=np.linalg.norm(beam_vec)
    if L==0: return
    unit=beam_vec/L
    pts=np.stack((XX,YY),axis=-1).astype(np.float32)
    vecs=pts-np.array(nc,np.float32)[None,None,:]
    proj=(vecs@unit).clip(0,L)
    tmap=proj/L
    alpha_map=(1-tmap)*alpha
    for c in range(3):
        reg=img[y:y+hb,x:x+wb,c].astype(np.float32)
        reg[inside]=reg[inside]*(1-alpha_map[inside])+color[c]*alpha_map[inside]
        img[y:y+hb,x:x+wb,c]=np.clip(reg,0,255).astype(np.uint8)

def draw_cell_cuboid(img,nc,fc,color,alpha,apply=True):
    faces=[
        [nc[0],nc[1],nc[2],nc[3]],
        [fc[0],fc[1],fc[2],fc[3]],
        [nc[0],nc[3],fc[3],fc[0]],
        [nc[1],nc[2],fc[2],fc[1]],
        [nc[0],nc[1],fc[1],fc[0]],
        [nc[3],nc[2],fc[2],fc[3]],
    ]
    if apply:
        c_near=tuple(np.mean(nc,axis=0).astype(int))
        c_far =tuple(np.mean(fc,axis=0).astype(int))
        for face in faces:
            poly=np.array(face,np.int32).reshape(-1,1,2)
            fill_gradient_polygon(img,poly,c_near,c_far,color,alpha)

def add_glow_effect(img, near, far,
                    glow_color=(0,255,255), glow_alpha=0.4,
                    dilate_iter=30, blur_sizes=(15,31,61), weights=(0.6,0.3,0.1)):
    pts=np.array(near+far,np.int32)
    hull=cv2.convexHull(pts)
    mask=np.zeros(img.shape[:2],np.uint8)
    cv2.fillPoly(mask,[hull],255)
    kernel=np.ones((3,3),np.uint8)
    md=cv2.dilate(mask,kernel,iterations=dilate_iter)
    norm=np.zeros_like(md,np.float32)/255
    for k,w in zip(blur_sizes,weights):
        bk=(k//2)*2+1
        b=cv2.GaussianBlur(md,(bk,bk),0).astype(np.float32)/255
        norm+=w*b
    norm=np.clip(norm,0,1)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    alpha_map=(norm*glow_alpha*(1-gray))[:,:,None]
    glow_layer=img.astype(np.float32)*0.3 + np.array(glow_color,np.float32)*0.7
    out=img.astype(np.float32)+glow_layer*alpha_map
    return np.clip(out,0,255).astype(np.uint8)

def get_cell_detection_classes(poly,dets,conf_th=0.5,shrink=0.5):
    x,y,wb,hb=cv2.boundingRect(poly)
    cell=(x,y,x+wb,y+hb)
    classes=set()
    for x1,y1,x2,y2,conf,cls in dets:
        if conf<conf_th: continue
        cx,cy=(x1+x2)/2,(y1+y2)/2
        w2,h2=(x2-x1)*shrink,(y2-y1)*shrink
        nx1,ny1=cx-w2/2,cy-h2/2
        nx2,ny2=cx+w2/2,cy+h2/2
        ix1,iy1=max(cell[0],nx1),max(cell[1],ny1)
        ix2,iy2=min(cell[2],nx2),min(cell[3],ny2)
        if ix1<ix2 and iy1<iy2:
            classes.add(int(cls))
    return classes

def draw_full_cuboid_beam_mesh_with_detection(img,near_q,far_q,dets,
                                              cols,rows,color,
                                              alpha,sign_alpha):
    nv=compute_grid_vertices(near_q,cols,rows)
    fv=compute_grid_vertices(far_q,cols,rows)
    car_cells=[]
    for i in range(rows):
        for j in range(cols):
            nc=[nv[i][j],nv[i][j+1],nv[i+1][j+1],nv[i+1][j]]
            fc=[fv[i][j],fv[i][j+1],fv[i+1][j+1],fv[i+1][j]]
            hull=cv2.convexHull(np.array(nc+fc,np.int32))
            cls_set=get_cell_detection_classes(hull,dets)
            if 0 in cls_set:
                car_cells.append(hull)
                draw_cell_cuboid(img,nc,fc,color,alpha,apply=False)
            elif 2 in cls_set:
                draw_cell_cuboid(img,nc,fc,color,sign_alpha,apply=True)
            else:
                draw_cell_cuboid(img,nc,fc,color,alpha,apply=True)
    return car_cells

# --------------------------
# Video & beam definitions
# --------------------------
youtube_url='https://www.youtube.com/watch?v=jOgVW9tue04&t=105s'
with yt_dlp.YoutubeDL({'format':'best','quiet':True}) as ydl:
    info=ydl.extract_info(youtube_url,download=False)
cap=cv2.VideoCapture(info['url'])

ret,frame=cap.read()
if not ret: raise IOError("No frames")
h,w=frame.shape[:2]
cap.set(cv2.CAP_PROP_POS_FRAMES,0)

# Matrix headlight beam polygon definitions for 2 beams
beam1_near=[(int(w*0.3625),int(h*0.75)),(int(w*0.4375),int(h*0.75)),
            (int(w*0.4375),int(h*0.82)),(int(w*0.3625),int(h*0.82))]
beam1_far =[ (int(w*0.20),int(h*0.30)),(int(w*0.50),int(h*0.30)),
             (int(w*0.50),int(h*0.60)),(int(w*0.20),int(h*0.60)) ]
beam2_near=[(int(w*0.5625),int(h*0.75)),(int(w*0.6375),int(h*0.75)),
            (int(w*0.6375),int(h*0.82)),(int(w*0.5625),int(h*0.82))]
beam2_far =[ (int(w*0.50),int(h*0.30)),(int(w*0.80),int(h*0.30)),
             (int(w*0.80),int(h*0.60)),(int(w*0.50),int(h*0.60)) ]

cols,rows=5,5
beam_color=(255,255,255)
default_alpha=0.6
LOWBEAM_ALPHA    = 0.1      # mesh when weather → low-beam
default_glow_a   = 0.40      # original halo strength
LOWBEAM_GLOW_A   = 0.1      # ← new: halo when in low-beam
sign_alpha=0.001
glow_color=(255,255,255)

bev_size=(300,300)
src_pts=np.float32([[w*0.3,h*0.6],[w*0.7,h*0.6],[w,h],[0,h]])
dst_pts=np.float32([[0,0],[bev_size[0],0],[bev_size[0],bev_size[1]],[0,bev_size[1]]])

# Threshold for switching to LOW-beam if strong adverse weather is detected
WEATHER_THR = 0.70

# -------------------------------------------------------------------
# Video output writer setup
# -------------------------------------------------------------------
fps = 5.0
output_path = "processed_output_6.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_size = (w + bev_size[0], max(h, bev_size[1]))
out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

# Hysteresis: require X frames of state change before toggling beams
HYST_FRAMES   = 5
highbeam_state = True
hb_on_count   = 0
hb_off_count  = 0

# --------------------------
# Main processing loop
# --------------------------
while True:
    ret,frame=cap.read()
    if not ret: break

    # 1) Run weather classifiers
    sp = detect_snow(frame, snow_model, device)
    rp = detect_rain(frame, rain_model, device)
    fp = detect_fog(frame,  fog_model,  device)

    # Decide if high-beams are off due to weather
    highbeam_off = (sp['snow'] > WEATHER_THR) or \
                   (rp['rain'] > WEATHER_THR) or \
                   (fp['fog']  > WEATHER_THR)

    # 2) Run segmentation (UNet) with weather-specific preprocessing
    mask = process_frame_unet(frame, unet_model, device,
                              snow_prob=sp['snow'],
                              rain_prob=rp['rain'],
                              fog_prob=fp['fog'])
    cm         = colorize_mask(mask)
    cm_resized = cv2.resize(cm,(w,h),interpolation=cv2.INTER_NEAREST)
    seg_ov     = cv2.addWeighted(frame,0.6,cm_resized,0.4,0)

    # 3) Bird's-eye view transformation
    bev=get_bird_eye_view(cm_resized,src_pts,dst_pts,bev_size)

    # 4) YOLO object detection
    dets=yolo_model(frame).xyxy[0].cpu().numpy()
    if dets.shape[0] > 0:
        dets[:, 5] = 0  # force all detections to "car" class (for demo)

    # 5) Matrix beam mesh simulation
    overlay = seg_ov.copy()
    curr_alpha = LOWBEAM_ALPHA if highbeam_off else default_alpha

    cars1 = draw_full_cuboid_beam_mesh_with_detection(
        overlay, beam1_near, beam1_far, dets,
        cols, rows,
        beam_color,
        curr_alpha,          # normal cells
        sign_alpha           # sign-suppressed cells
    )
    cars2 = draw_full_cuboid_beam_mesh_with_detection(
        overlay, beam2_near, beam2_far, dets,
        cols, rows,
        beam_color,
        curr_alpha,
        sign_alpha
    )

    # 6) Glow effect for beams (brighter in high-beam mode)
    glow_a = LOWBEAM_GLOW_A if highbeam_off else default_glow_a
    overlay = add_glow_effect(
        overlay, beam1_near, beam1_far,
        glow_color, glow_alpha=glow_a
    )
    overlay = add_glow_effect(
        overlay, beam2_near, beam2_far,
        glow_color, glow_alpha=glow_a
    )

    # 7) Restore segmentation where cars are detected (overwrites beam for cars)
    kernel=np.ones((3,3),np.uint8)
    for poly in cars1+cars2:
        m=np.zeros((h,w),np.uint8)
        cv2.fillPoly(m,[poly],255)
        md=cv2.dilate(m,kernel,iterations=55)
        mk3=cv2.merge([md,md,md])
        overlay=np.where(mk3==255,seg_ov,overlay)

    # 8) Combine overlay and BEV for display/output
    combined=np.zeros((max(h,bev_size[1]),w+bev_size[0],3),dtype=np.uint8)
    combined[:h,:w]=overlay
    combined[:bev_size[1],w:w+bev_size[0]]=bev

    # 9) Annotate rain prediction
    if rp['rain']>rp['no_rain']:
        rl,rc='Rain',rp['rain']
    else:
        rl,rc='No Rain',rp['no_rain']
    cv2.putText(combined,f"{rl} ({rc*100:.1f}%)",(10,combined.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255) if rl=='Rain' else (0,200,0),2)

    # 10) Annotate fog prediction
    fp=detect_fog(frame,fog_model,device)
    if fp['fog']>fp['no_fog']:
        fl,fc='Fog',fp['fog']
    else:
        fl,fc='No Fog',fp['no_fog']
    cv2.putText(combined,f"{fl} ({fc*100:.1f}%)",(10,combined.shape[0]-35),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,200,0) if fl=='Fog' else (0,200,200),2)

    # 11) Annotate night/day prediction
    npd=detect_night(frame,night_model,device)
    if npd['night']>npd['day']:
        nl,nc='Night',npd['night']
    else:
        nl,nc='Day',npd['day']
    cv2.putText(combined,f"{nl} ({nc*100:.1f}%)",(10,combined.shape[0]-60),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,0,255) if nl=='Night' else (255,255,255),2)

    # 12) Annotate snow prediction
    if sp['snow']>sp['no_snow']:
        sl,sc='Snow',sp['snow']
    else:
        sl,sc='No Snow',sp['no_snow']
    cv2.putText(combined,f"{sl} ({sc*100:.1f}%)",(10,combined.shape[0]-85),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0) if sl=='Snow' else (200,200,200),2)

    # 13) Show result and write to output video
    cv2.imshow("Seg+Beam+BEV+Rain+Fog+Night+Snow", combined)
    out.write(combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --------------------------
# Cleanup: release all resources
# --------------------------
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Saved processed video to {output_path}")
