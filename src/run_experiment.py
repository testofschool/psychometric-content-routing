#!/usr/bin/env python3
"""
Psychometric Content Routing - Groq/Llama 70B Experiment
Run: GROQ_API_KEY=gsk_... python src/run_experiment.py
"""
import csv,hashlib,json,os,re,sys,time
import numpy as np
from pathlib import Path
from scipy.special import expit as sigmoid
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/"data"; PROMPTS=ROOT/"prompts"; CACHE_FILE=ROOT/"cache"/"api_cache.json"
RESULTS=ROOT/"results"; RESULTS.mkdir(exist_ok=True)
MODEL="llama-3.3-70b-versatile"; JUDGE_MODEL="llama-3.1-8b-instant"
BUDGET=4; JUDGE_REPEATS=3; DELAY=2.5; SEED=42
np.random.seed(SEED)

def load_corpus():
    rows=[]
    with open(DATA/"passages.csv") as f:
        for r in csv.DictReader(f): r["true_difficulty"]=float(r["true_difficulty"]); rows.append(r)
    return rows

def load_users():
    rows=[]
    with open(DATA/"users.csv") as f:
        for r in csv.DictReader(f): r["theta"]=float(r["theta"]); rows.append(r)
    return rows

def frontier_value(theta,b,a=1.5):
    p=sigmoid(a*(theta-b)); return p*(1-p)

def route_corpus_prefix(user,corpus,k): return list(range(k))
def route_random(user,corpus,k):
    return sorted(np.random.RandomState(SEED).choice(len(corpus),k,replace=False).tolist())
def route_difficulty_only(user,corpus,k):
    return sorted(np.argsort([c["true_difficulty"] for c in corpus])[-k:].tolist())
def route_pcr(user,corpus,k):
    fvs=[frontier_value(user["theta"],c["true_difficulty"]) for c in corpus]
    return sorted(np.argsort(fvs)[-k:].tolist())

def route_pcr_lltm(user,corpus,k):
    """PCR using LLTM-estimated difficulty (calibrated to theta scale). Use --lltm flag."""
    fvs=[frontier_value(user["theta"],float(c.get("lltm_calibrated",c["true_difficulty"]))) for c in corpus]
    return sorted(np.argsort(fvs)[-k:].tolist())

USE_LLTM = "--lltm" in sys.argv
STRATEGIES={"corpus_prefix":route_corpus_prefix,"random":route_random,
            "difficulty_only":route_difficulty_only,
            "pcr": route_pcr_lltm if USE_LLTM else route_pcr}
if USE_LLTM: print("MODE: LLTM-estimated difficulty (rho~0.50)")
else: print("MODE: Author-assigned difficulty (upper bound)")

CACHE={}
if CACHE_FILE.exists():
    with open(CACHE_FILE) as f: CACHE=json.load(f)
def save_cache():
    CACHE_FILE.parent.mkdir(exist_ok=True)
    with open(CACHE_FILE,"w") as f: json.dump(CACHE,f,indent=2)

def call_groq(system,user_msg,model=MODEL,temperature=0.3,max_tokens=550,salt=""):
    key=hashlib.md5(f"{model}:{system}:{user_msg}:{temperature}:{salt}".encode()).hexdigest()
    if key in CACHE: return CACHE[key]
    api_key=os.environ.get("GROQ_API_KEY","")
    if not api_key: print("ERROR: Set GROQ_API_KEY"); sys.exit(1)
    from groq import Groq; client=Groq(api_key=api_key)
    for attempt in range(3):
        try:
            time.sleep(DELAY)
            resp=client.chat.completions.create(model=model,
                messages=[{"role":"system","content":system},{"role":"user","content":user_msg}],
                temperature=temperature,max_tokens=max_tokens)
            result={"ok":True,"content":resp.choices[0].message.content,
                    "tokens":resp.usage.total_tokens if resp.usage else 0}
            CACHE[key]=result; save_cache(); return result
        except Exception as e:
            print(f"  Retry {attempt+1}: {e}"); time.sleep(2**(attempt+1))
    result={"ok":False,"content":"","tokens":0}; CACHE[key]=result; save_cache(); return result

def parse_json(text):
    try: return json.loads(re.sub(r"```json|```","",text).strip())
    except:
        nums=re.findall(r"\d+\.?\d*",text[:50])
        return {"score":float(nums[0])} if nums else None

def phase1(corpus):
    print("\n=== Phase 1: Difficulty Validation ===")
    system=(PROMPTS/"phase1_difficulty_rating.txt").read_text().split("USER:")[0].replace("SYSTEM:","").strip()
    ratings=[]
    for c in corpus:
        print(f"  {c['id']}...",end=" ",flush=True)
        resp=call_groq(system,f"Rate this passage:\n\n{c['text']}")
        p=parse_json(resp["content"]) if resp["ok"] else None
        r=p.get("difficulty",5) if p else 5; ratings.append(r); print(f"LLM={r}")
    rho,_=spearmanr([c["true_difficulty"] for c in corpus],ratings)
    print(f"  rho={rho:.3f}"); return {"ratings":ratings,"rho":float(rho)}

def phase2(corpus,users):
    print("\n=== Phase 2: Generation ===")
    tmpl=(PROMPTS/"phase2_generation.txt").read_text()
    results={}
    for u in users:
        results[u["id"]]={}
        sys_p=tmpl.split("USER:")[0].replace("SYSTEM:","").strip().replace("{user_label}",u["label"])
        for sn,sf in STRATEGIES.items():
            idx=sf(u,corpus,BUDGET); chunks=[corpus[i] for i in idx]
            ct="\n\n".join(f"Passage {j+1}: {c['text']}" for j,c in enumerate(chunks))
            print(f"  {u['id']:>10} x {sn:<18}",end=" ",flush=True)
            resp=call_groq(sys_p,f"Please explain:\n\n{ct}"); print(f"{resp['tokens']}tok")
            results[u["id"]][sn]={"selected":idx,"response":resp["content"][:2000],"tokens":resp["tokens"]}
    return results

def phase3(users,gen):
    print("\n=== Phase 3: Pairwise Judge ===")
    pairs=[("pcr","difficulty_only"),("pcr","random")]; results={}
    for pa,pb in pairs:
        pn=f"{pa}_vs_{pb}"; results[pn]={"wins_a":0,"total":0}
        for u in users:
            ea=gen[u["id"]][pa]["response"][:1800]; eb=gen[u["id"]][pb]["response"][:1800]
            for rep in range(JUDGE_REPEATS):
                swap=(rep%2==1); a_t,b_t=(eb,ea) if swap else (ea,eb)
                sys_p=(PROMPTS/"phase3_pairwise_judge.txt").read_text()
                sys_p=sys_p.replace("{user_label}",u["label"]).replace("{explanation_a}",a_t).replace("{explanation_b}",b_t)
                s=sys_p.split("USER:")[0].replace("SYSTEM:","").strip()
                um=sys_p.split("USER:")[-1].strip() if "USER:" in sys_p else sys_p
                print(f"  {u['id']:>10} {pn} r{rep}",end=" ",flush=True)
                resp=call_groq(s,um,model=JUDGE_MODEL,temperature=0.1,max_tokens=180,salt=f"p3:{u['id']}:{pn}:{rep}")
                p=parse_json(resp["content"]) if resp["ok"] else None
                w=p.get("winner","?") if p else "?"
                if swap: w="B" if w=="A" else "A" if w=="B" else "?"
                if w=="A": results[pn]["wins_a"]+=1
                results[pn]["total"]+=1; print(f"w={w}")
    return results

def phase4(users,gen):
    print("\n=== Phase 4: Absolute Judge ===")
    scores={s:[] for s in STRATEGIES}
    for u in users:
        for sn in STRATEGIES:
            exp=gen[u["id"]][sn]["response"][:1800]
            for rep in range(JUDGE_REPEATS):
                sys_p=(PROMPTS/"phase4_absolute_judge.txt").read_text()
                sys_p=sys_p.replace("{user_label}",u["label"]).replace("{explanation}",exp)
                s=sys_p.split("USER:")[0].replace("SYSTEM:","").strip()
                um=sys_p.split("USER:")[-1].strip() if "USER:" in sys_p else sys_p
                print(f"  {u['id']:>10} {sn:<18} r{rep}",end=" ",flush=True)
                resp=call_groq(s,um,model=JUDGE_MODEL,temperature=0.1,max_tokens=180,salt=f"p4:{u['id']}:{sn}:{rep}")
                p=parse_json(resp["content"]) if resp["ok"] else None
                sc=p.get("score") if p else None
                if sc is not None: scores[sn].append(float(sc))
                print(f"s={sc}")
    return scores

def main():
    corpus=load_corpus(); users=load_users()
    print(f"Corpus: {len(corpus)}, Users: {len(users)}, Budget: {BUDGET}")
    p1=phase1(corpus); p2=phase2(corpus,users); p3=phase3(users,p2); p4=phase4(users,p2)
    print("\n"+"="*60)
    print("PSYCHOMETRIC CONTENT ROUTING - RESULTS")
    print("="*60)
    print(f"\nPhase 1: LLM difficulty vs true: rho = {p1['rho']:.3f}")
    print(f"\nPhase 2+4: Equal-budget scores (4 chunks each)")
    print(f"  {'Strategy':<20} {'Mean+/-Std':>12}")
    for s in STRATEGIES:
        v=p4[s]
        if v: print(f"  {s:<20} {np.mean(v):.2f}+/-{np.std(v):.2f}")
    print(f"\nPhase 3: Pairwise")
    for pair,res in p3.items():
        pct=res["wins_a"]/max(res["total"],1)*100
        print(f"  {pair}: PCR wins {res['wins_a']}/{res['total']} ({pct:.0f}%)")
    all_res={"phase1":p1,"phase3":p3,"phase4":{s:{"mean":float(np.mean(v)),"std":float(np.std(v))} for s,v in p4.items() if v}}
    with open(RESULTS/"experiment_results.json","w") as f: json.dump(all_res,f,indent=2,default=float)
    print(f"\nSaved to {RESULTS/'experiment_results.json'}")

if __name__=="__main__": main()
