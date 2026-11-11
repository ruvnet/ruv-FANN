"""Run paired PAN2 inference on a separately provisioned CUDA machine.

Requires the pinned, clean upstream checkout and complete native baseline and
candidate checkpoint directories. Does not score or promote a model.
"""
import argparse, hashlib, json, os, pathlib, shutil, signal, subprocess, time
PIN = "0fb907b6e16a82ead6d691d1de2eef6299c71e9a"

def digest(p):
    with p.open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()

def inventory(p):
    paths = sorted(p.rglob("*.safetensors"))
    if not paths: raise RuntimeError("checkpoint requires Safetensors shards")
    return {str(f.relative_to(p)): digest(f) for f in paths}

def main():
    a = argparse.ArgumentParser()
    for name in ("repo", "baseline", "candidate", "prompts", "output"):
        a.add_argument("--"+name, required=True, type=pathlib.Path)
    a.add_argument("--seed",type=int,default=123)
    a.add_argument("--timeout",type=int,default=7200)
    args = a.parse_args()
    for name in ("repo","baseline","candidate","prompts","output"):
        setattr(args,name,getattr(args,name).resolve())
    if not shutil.which("nvidia-smi"): raise RuntimeError("CUDA executor unavailable: nvidia-smi missing")
    subprocess.run(["nvidia-smi"],check=True,stdout=subprocess.DEVNULL)
    head = subprocess.check_output(["git","-C",str(args.repo),"rev-parse","HEAD"],text=True).strip()
    if head != PIN: raise RuntimeError("PAN2 checkout does not match pinned revision")
    if subprocess.check_output(["git","-C",str(args.repo),"status","--porcelain"],text=True).strip():
        raise RuntimeError("PAN2 checkout must be clean")
    if args.timeout<1 or args.baseline==args.candidate: raise RuntimeError("invalid timeout or identical checkpoint directories")
    before = {"baseline":inventory(args.baseline),"candidate":inventory(args.candidate)}
    if before["baseline"]==before["candidate"]: raise RuntimeError("candidate has no weight differences")
    prompts_hash = digest(args.prompts)
    args.output.mkdir(parents=True,exist_ok=False)
    prompts = args.output / "prompts.json"
    shutil.copyfile(args.prompts,prompts)
    # Fixed sampler; remove inherited shell EXTRA_ARGS and sampler overrides.
    env = os.environ.copy()
    env.update({"PAN_ROOT":str(args.repo),"SEED":str(args.seed),"NUM_GPUS":"1","NUM_STEPS":"50","FLOW_SHIFT":"9","GUIDANCE_SCALE":"2.5","VIDEO_HEIGHT":"704","VIDEO_WIDTH":"1248","FPS":"24","VIDEO_LENGTH":"241","TASK":"auto","EXTRA_ARGS":"","TRAINING_CONFIG":"config/sft_29B_720p.yaml","NEGATIVE_PROMPT_FILE":"inference/prompts/negative_prompt.json","HF_HUB_OFFLINE":"1"})
    runs = {}
    for label in ("baseline","candidate"):
        start = time.perf_counter()
        with (args.output / (label+".log")).open("w") as log:
            process = subprocess.Popen(["bash",str(args.repo/"inference/inference.sh"),str(getattr(args,label)),str(args.output/label),str(prompts)],cwd=args.repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code = process.wait(timeout=args.timeout)
                if code: raise RuntimeError(f"{label} inference failed: exit {code}")
            except BaseException:
                try: os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait()
                raise
        videos = sorted((args.output/label).rglob("*.mp4"))
        if not videos: raise RuntimeError("inference produced no videos")
        runs[label] = {"seconds":time.perf_counter()-start,"videos":{str(p.relative_to(args.output/label)):digest(p) for p in videos}}
    after = {"baseline":inventory(args.baseline),"candidate":inventory(args.candidate)}
    if before!=after or digest(prompts)!=prompts_hash: raise RuntimeError("inputs changed during evaluation")
    result = {"schema":"ruv-fann.pan2-paired-inference/1","pan2_commit":head,"seed":args.seed,"prompts_sha256":prompts_hash,"checkpoints":before,"runs":runs,"behavioral_validation":"requires independent scoring","promotion":"not_performed"}
    (args.output/"comparison.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__ == "__main__": main()
