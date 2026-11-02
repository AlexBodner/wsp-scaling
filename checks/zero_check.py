import json,urllib.request,warnings,logging,torch,transformers
from collections import Counter
warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
from transformers import AutoConfig,AutoModelForCausalLM
def raw(m):
    try:
        with urllib.request.urlopen(urllib.request.Request(f'https://huggingface.co/{m}/raw/main/config.json',headers={'User-Agent':'x'}),timeout=25) as r:return json.load(r)
    except Exception as e:return {'ERR':str(e)[:80]}
print('Qwen3.8 ids'); 
with urllib.request.urlopen(urllib.request.Request('https://huggingface.co/api/models?author=Qwen&search=Qwen3.8&limit=30',headers={'User-Agent':'x'}),timeout=25) as r: print([m['id'] for m in json.load(r)])
c=raw('Qwen/Qwen3.8-27B'); t=c.get('text_config',c)
print('3.8-27B',c.get('model_type'),t.get('model_type'),{k:t.get(k) for k in ['vocab_size','hidden_size','num_hidden_layers','num_attention_heads','intermediate_size','hidden_act','tie_word_embeddings']},str(t.get('layer_types'))[:90])
def counts(cfg):
    with torch.device('meta'): m=AutoModelForCausalLM.from_config(cfg)
    tot=sum(p.numel() for p in m.parameters()); emb=sum(p.numel() for n,p in m.named_parameters() if any(k in n for k in ['wte','wpe','embed','lm_head'])); return m,tot,emb
for name in ['roneneldan/TinyStories-1M','roneneldan/TinyStories-8M','EleutherAI/pythia-14m','HuggingFaceTB/SmolLM-135M','gpt2','Qwen/Qwen3.5-0.8B','Qwen/Qwen3.8-27B']:
    try:
        cfg=AutoConfig.from_pretrained(name)
        if hasattr(cfg,'text_config') and cfg.text_config is not None: cfg=cfg.text_config
        m,tot,emb=counts(cfg)
        norms=Counter(type(x).__name__ for x in m.modules() if 'norm' in type(x).__name__.lower())
        print(f'{name}: params {tot:,}; embedding share {emb/tot:.1%}; norms {dict(norms)}; layers {getattr(cfg,"num_hidden_layers",getattr(cfg,"num_layers",getattr(cfg,"n_layer",None)))}')
        # zero-state check on the same code with a reduced config, so it is CPU light
        small=cfg.__class__.from_dict(cfg.to_dict())
        small.vocab_size=1000
        for k in ('num_hidden_layers','n_layer','num_layers'):
            if hasattr(small,k): setattr(small,k,min(2,getattr(small,k)))
        if hasattr(small,'layer_types') and small.layer_types: small.layer_types=small.layer_types[:2]
        if hasattr(small,'attention_layers') and small.attention_layers: small.attention_layers=small.attention_layers[:2]
        z=AutoModelForCausalLM.from_config(small).eval()
        with torch.no_grad():
            for p in z.parameters(): p.zero_()
            ids=torch.randint(0,1000,(1,8)); o=z(input_ids=ids).logits
        print(f'   zero weights (2 layers, vocab 1000): logits finite={bool(o.isfinite().all())} max|logit|={float(o.nan_to_num().abs().max()):.3g}')
    except Exception as e:
        print(name,'FAILED',type(e).__name__,str(e)[:180])
