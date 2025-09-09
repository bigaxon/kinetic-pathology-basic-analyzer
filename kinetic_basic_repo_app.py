#!/usr/bin/env python3
import os, cv2, time, math, uuid, tempfile, numpy as np, gradio as gr, matplotlib.pyplot as plt
from collections import deque, defaultdict
from typing import List, Tuple, Dict
from scipy.optimize import linear_sum_assignment
from skimage import filters, measure, morphology, restoration, exposure

TVW = 0.03
MINA = 5
MAXA = 150
OPENR = 0
LINKD = 30.0
TRAIL = 15
RADC = 2
DT_MIN = 1.5

def _seg_primary(imgf: np.ndarray):
    den = restoration.denoise_tv_chambolle(imgf, weight=TVW, channel_axis=None)
    den = exposure.equalize_adapthist(den, clip_limit=0.01)
    if OPENR > 0:
        den = morphology.opening(den, morphology.disk(OPENR))
    t = filters.threshold_otsu(den)
    return den > t

def _seg_fallback(imgf: np.ndarray):
    den = exposure.equalize_adapthist(imgf, clip_limit=0.005)
    local = den > filters.threshold_local(den, block_size=31, offset=-0.01)
    bw = morphology.remove_small_holes(local, area_threshold=16)
    bw = morphology.remove_small_objects(bw, MINA)
    bw = morphology.opening(bw, morphology.disk(1))
    return bw

def segment(gray: np.ndarray) -> Tuple[np.ndarray, List[Tuple[float, float]], List[float]]:
    img = gray.astype(np.float32)
    if img.max() > 1.5:
        img = img/255.0
    bw = _seg_primary(img)
    lbl = measure.label(bw, connectivity=2)
    props = measure.regionprops(lbl)
    good=set(); cents=[]; areas=[]
    for p in props:
        if MINA <= p.area <= MAXA:
            good.add(p.label); cy, cx = p.centroid; cents.append((float(cx), float(cy))); areas.append(float(p.area))
    if len(cents) < 5:
        bw2 = _seg_fallback(img)
        lbl2 = measure.label(bw2, connectivity=2)
        props2 = measure.regionprops(lbl2)
        good2=set(); cents2=[]; areas2=[]
        for p in props2:
            if MINA <= p.area <= MAXA:
                good2.add(p.label); cy, cx = p.centroid; cents2.append((float(cx), float(cy))); areas2.append(float(p.area))
        if len(cents2) > len(cents):
            return (np.isin(lbl2, list(good2))).astype(np.uint8), cents2, areas2
    return (np.isin(lbl, list(good))).astype(np.uint8), cents, areas

def eucl(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    return float(math.hypot(a[0]-b[0], a[1]-b[1]))

def link(prev, curr, maxd):
    if not prev and not curr: return [], [], []
    if not prev: return [], list(range(len(curr))), []
    if not curr: return [], [], list(range(len(prev)))
    cost = np.zeros((len(prev), len(curr)), dtype=np.float32)
    for i,p in enumerate(prev):
        for j,c in enumerate(curr):
            cost[i,j]=eucl(p,c)
    ri, ci = linear_sum_assignment(cost)
    matches=[]; usedp=set(); usedc=set()
    for i,j in zip(ri,ci):
        if cost[i,j] <= maxd:
            matches.append((i,j)); usedp.add(i); usedc.add(j)
    relax = 1.5*maxd
    for i in range(len(prev)):
        if i in usedp: continue
        bestj=-1; bestd=float('inf')
        for j in range(len(curr)):
            if j in usedc: continue
            d=cost[i,j]
            if d<bestd and d<=relax:
                bestd=d; bestj=j
        if bestj>=0:
            matches.append((i,bestj)); usedp.add(i); usedc.add(bestj)
    newc=[j for j in range(len(curr)) if j not in usedc]
    deadp=[i for i in range(len(prev)) if i not in usedp]
    return matches, newc, deadp

def color_for_id(idx:int)->Tuple[int,int,int]:
    h=(idx*47)%180; s,v=200,255
    hsv=np.uint8([[[h,s,v]]]); bgr=cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0].tolist()
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def analyze_video(video_path: str, out_dir: str):
    t0=time.time()
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise RuntimeError("Could not open video.")
    fps=cap.get(cv2.CAP_PROP_FPS); w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps<=0 or np.isnan(fps): fps=10.0
    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    overlay_path=os.path.join(out_dir, f"overlay_{uuid.uuid4().hex}.mp4")
    writer=cv2.VideoWriter(overlay_path, fourcc, fps, (w,h))

    next_id=1
    tpos: Dict[int,List[Tuple[int,float,float]]]=defaultdict(list)
    trails: Dict[int,deque]=defaultdict(lambda: deque(maxlen=TRAIL))
    colors: Dict[int,Tuple[int,int,int]]={}
    prev_c=[]; prev_ids=[]
    all_areas=[]

    fidx=0
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, curr_c, curr_a = segment(gray)
        matches, births, _ = link(prev_c, curr_c, LINKD)
        curr_ids=[-1]*len(curr_c)
        for ip,jc in matches:
            curr_ids[jc]=prev_ids[ip]
        for j in births:
            tid=next_id; next_id+=1; curr_ids[j]=tid; colors[tid]=color_for_id(tid)
        for j,(x,y) in enumerate(curr_c):
            tid=curr_ids[j]
            if tid<0: continue
            tpos[tid].append((fidx,float(x),float(y)))
            trails[tid].append((float(x),float(y)))
            if j < len(curr_a):
                all_areas.append(float(curr_a[j]))
        ov=frame.copy()
        for j,(x,y) in enumerate(curr_c):
            tid=curr_ids[j]
            if tid<0: continue
            col=colors.get(tid,(255,255,255))
            cx,cy=int(round(x)),int(round(y))
            cv2.circle(ov,(cx,cy),RADC,col,-1)
            pts=list(trails[tid])
            for a in range(1,len(pts)):
                p1=(int(round(pts[a-1][0])), int(round(pts[a-1][1])))
                p2=(int(round(pts[a][0])), int(round(pts[a][1])))
                cv2.line(ov,p1,p2,col,1)
            cv2.putText(ov,f"{tid}",(cx+4,cy-4),cv2.FONT_HERSHEY_SIMPLEX,0.35,col,1,cv2.LINE_AA)
        writer.write(ov)
        prev_c=curr_c; prev_ids=curr_ids; fidx+=1

    cap.release(); writer.release()

    all_inst=[]; path_lens=[]; disps=[]; persist=[]; per_rel=[]; max_steps=0
    for tid, pts in tpos.items():
        if len(pts)<2:
            per_rel.append([0.0]); max_steps=max(max_steps,1)
            path_lens.append(0.0); disps.append(0.0); persist.append(0.0); continue
        pts_sorted=sorted(pts,key=lambda z:z[0])
        xs=[p[1] for p in pts_sorted]; ys=[p[2] for p in pts_sorted]
        step_ds=[]
        for a in range(1,len(xs)):
            d=math.hypot(xs[a]-xs[a-1], ys[a]-ys[a-1]); step_ds.append(d); all_inst.append(d/DT_MIN)
        pl=float(sum(step_ds)); disp=float(math.hypot(xs[-1]-xs[0], ys[-1]-ys[0]))
        path_lens.append(pl); disps.append(disp); persist.append((disp/pl) if pl>1e-8 else 0.0)
        rel=[0.0]; x0,y0=xs[0],ys[0]
        for a in range(1,len(xs)):
            rel.append(float(math.hypot(xs[a]-x0, ys[a]-y0)))
        per_rel.append(rel); max_steps=max(max_steps,len(rel))

    total_cells = len(tpos)
    if len(all_areas)>0:
        mean_area = float(np.mean(all_areas))
        eq_diam = 2.0*np.sqrt(max(mean_area,1e-8)/np.pi)
        mig_thresh = 2.0*eq_diam
    else:
        mig_thresh = 5.0

    disps_arr = np.array(disps, dtype=float)
    mig_mask = disps_arr >= mig_thresh
    migrating_cells = int(np.sum(mig_mask))
    pct_migrating = (100.0*migrating_cells/total_cells) if total_cells>0 else 0.0

    mean_speed = float(np.mean(all_inst)) if len(all_inst)>0 else 0.0
    sd_speed = float(np.std(all_inst)) if len(all_inst)>0 else 0.0

    # Min speed from migrating cells only
    per_track_speeds = []
    for tid, pts in tpos.items():
        if len(pts) < 2:
            per_track_speeds.append([]); continue
        pts_sorted = sorted(pts, key=lambda z:z[0])
        xs=[p[1] for p in pts_sorted]; ys=[p[2] for p in pts_sorted]
        ts=[]
        for a in range(1,len(xs)):
            d=math.hypot(xs[a]-xs[a-1], ys[a]-ys[a-1])
            ts.append(d/DT_MIN)
        per_track_speeds.append(ts)
    mig_speeds = []
    for k, is_mig in enumerate(mig_mask):
        if is_mig: mig_speeds.extend(per_track_speeds[k])
    min_speed = float(np.min(mig_speeds)) if len(mig_speeds)>0 else 0.0

    max_speed = float(np.max(all_inst)) if len(all_inst)>0 else 0.0
    path_arr = np.array(path_lens, dtype=float)
    mean_path = float(np.mean(path_arr)) if path_arr.size>0 else 0.0
    sd_path = float(np.std(path_arr)) if path_arr.size>0 else 0.0
    mean_disp_all = float(np.mean(disps_arr)) if disps_arr.size>0 else 0.0
    sd_disp = float(np.std(disps_arr)) if disps_arr.size>0 else 0.0

    migratory_index = (pct_migrating/100.0) * mean_speed

    # MSD-like mean displacement vs sqrt(time) for migrating, top 50% by path length
    keep_idx = np.where(mig_mask)[0]
    if keep_idx.size > 0:
        cutoff = np.percentile(path_arr[keep_idx], 50.0)
        keep_idx = keep_idx[path_arr[keep_idx] >= cutoff]
    mean_disp=[]; sqrt_t=[]
    for step in range(max_steps):
        vals=[]
        for k in keep_idx:
            rel = per_rel[k]
            if step < len(rel): vals.append(rel[step])
        mean_disp.append(float(np.mean(vals)) if len(vals)>0 else np.nan)
        t_min = step*DT_MIN; sqrt_t.append(math.sqrt(t_min))

    def fig_speed_hist():
        fig=plt.figure(figsize=(5,4))
        if len(all_inst)>0: plt.hist(all_inst,bins=30)
        plt.xlabel("Instantaneous velocity (px/min)"); plt.ylabel("Count"); plt.title("Instantaneous Velocity Distribution"); plt.tight_layout(); return fig

    def fig_persistence_hist():
        fig=plt.figure(figsize=(5,4))
        if len(persist)>0: plt.hist(persist,bins=30,range=(0,1))
        plt.xlabel("Persistence (displacement / path length)"); plt.ylabel("Cells"); plt.title("Persistence Distribution"); plt.tight_layout(); return fig

    def fig_pathlen_hist():
        fig=plt.figure(figsize=(5,4))
        if len(path_lens)>0: plt.hist(path_lens,bins=30)
        plt.xlabel("Path length (px)"); plt.ylabel("Cells"); plt.title("Path Length Distribution"); plt.tight_layout(); return fig

    def fig_displacement_hist():
        fig=plt.figure(figsize=(5,4))
        if len(disps)>0: plt.hist(disps,bins=30)
        plt.xlabel("Displacement (px)"); plt.ylabel("Cells"); plt.title("Displacement Distribution"); plt.tight_layout(); return fig

    def fig_md_vs_sqrttime():
        x=np.array(sqrt_t); y=np.array(mean_disp)
        msk=np.isfinite(x) & np.isfinite(y)
        x_=x[msk]; y_=y[msk]
        fig=plt.figure(figsize=(5,4))
        plt.plot(x_, y_, marker="o", linewidth=1)
        if len(x_)>=2 and np.any(x_>0):
            m,b = np.polyfit(x_, y_, 1)
            yhat=m*x_+b
            ss_res=np.sum((y_-yhat)**2); ss_tot=np.sum((y_-np.mean(y_))**2) if len(y_)>1 else 0.0
            r2=1.0 - (ss_res/ss_tot) if ss_tot>0 else 1.0
            xx=np.linspace(np.min(x_), np.max(x_), 100); yy=m*xx+b
            plt.plot(xx, yy, 'r-', linewidth=1.5)
            xpos = np.min(x_) + 0.05*(np.max(x_)-np.min(x_))
            ypos = np.max(y_) - 0.1*(np.max(y_)-np.min(y_))
            plt.text(xpos, ypos, f"$R^2$ = {r2:.3f}", fontsize=9)
        plt.xlabel("√time (min$^{1/2}$)"); plt.ylabel("Mean displacement (px)"); plt.title("Mean Displacement vs √time"); plt.tight_layout(); return fig

    def fig_origin():
        fig=plt.figure(figsize=(5,5)); ax=fig.add_subplot(111)
        for tid,pts in tpos.items():
            if len(pts)<2: continue
            pts_sorted=sorted(pts,key=lambda z:z[0])
            x0,y0=pts_sorted[0][1], pts_sorted[0][2]
            relx=[p[1]-x0 for p in pts_sorted]; rely=[p[2]-y0 for p in pts_sorted]
            ax.plot(relx,rely,linewidth=0.8)
        ax.set_aspect("equal", adjustable="box"); ax.set_xlabel("X (px)"); ax.set_ylabel("Y (px)"); ax.set_title("Trajectories from Origin"); ax.grid(True, linewidth=0.3); plt.tight_layout(); return fig

    summary = (
        f"### Research Summary\n"
        f"**POPULATION**\n\n"
        f"- Total Cells: {total_cells}\n"
        f"- Total Migrating Cells: {migrating_cells} ({pct_migrating:.1f}%)\n\n"
        f"**KINETICS**\n\n"
        f"- Mean Speed: {mean_speed:.2f} px/min (±{sd_speed:.2f})\n"
        f"- Min/Max Speed: {min_speed:.2f} / {max_speed:.2f} px/min\n"
        f"- Mean Path Length: {mean_path:.2f} px (±{sd_path:.2f})\n"
        f"- Mean Displacement: {mean_disp_all:.2f} px (±{sd_disp:.2f})\n\n"
        f"**SCORING**\n\n"
        f"- Migratory Index: {migratory_index:.2f}\n\n"
    )

    analysis_secs=time.time()-t0

    return {
        "overlay_path": overlay_path,
        "fig_speed": fig_speed_hist(),
        "fig_persistence": fig_persistence_hist(),
        "fig_pathlen": fig_pathlen_hist(),
        "fig_displacement": fig_displacement_hist(),
        "fig_md_vs_sqrttime": fig_md_vs_sqrttime(),
        "fig_origin": fig_origin(),
        "summary_md": summary,
        "analysis_secs": analysis_secs,
    }

def _run_app(video_file):
    if video_file is None:
        raise gr.Error("Please upload a video.")
    input_path = video_file
    workdir = tempfile.mkdtemp(prefix="kp_basic_")
    res = analyze_video(input_path, workdir)
    footer = f"analysis time: {res['analysis_secs']:.2f} s | RUO"
    return (
        input_path,
        res["overlay_path"],
        res["fig_speed"],
        res["fig_persistence"],
        res["fig_pathlen"],
        res["fig_displacement"],
        res["fig_md_vs_sqrttime"],
        res["fig_origin"],
        res["summary_md"],
        footer,
    )

def build_ui():
    with gr.Blocks(css="button[download]{display:none!important;}") as demo:
        gr.Markdown("# Kinetic Pathology – Basic Analyzer (RUO)")
        with gr.Row():
            video_in = gr.Video(label="Video (MP4/MOV)", sources=["upload"], interactive=True, include_audio=False)
            vid_overlay = gr.Video(label="Tracking Overlay", include_audio=False)

        run_btn = gr.Button("run analysis", variant="primary")

        gr.Markdown("### Distributions")
        with gr.Row():
            p_speed = gr.Plot(label="Instantaneous velocity")
            p_persist = gr.Plot(label="Persistence")
            p_pathlen = gr.Plot(label="Path length")
            p_disp = gr.Plot(label="Displacement")

        with gr.Row():
            p_md = gr.Plot(label="Mean displacement vs √time")
            p_origin = gr.Plot(label="Trajectories from origin")

        summary_box = gr.Markdown("")
        gr.Markdown(
            "### Reference  \n"
            "Carbonell, W. S., Murase, S.-I., Horwitz, A. F., & Mandell, J. W. (2005). "
            "*Migration of perilesional microglia after focal brain injury and modulation by CC chemokine receptor 5: "
            "An in situ time-lapse confocal imaging study.* Journal of Neuroscience, 25(30), 7040–7047. "
            "[Full text](https://www.jneurosci.org/content/25/30/7040)"
        )
        footer_text = gr.Markdown("")

        run_btn.click(
            fn=_run_app,
            inputs=[video_in],
            outputs=[video_in, vid_overlay, p_speed, p_persist, p_pathlen, p_disp, p_md, p_origin, summary_box, footer_text],
            show_progress=True
        )
    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
