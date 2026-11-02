import warnings,logging,torch
warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
from transformers import AutoConfig,AutoModelForCausalLM
for name in ['Qwen/Qwen3.5-0.8B','Qwen/Qwen3.8-27B']:
    c=AutoConfig.from_pretrained(name); c=c.text_config
    s=c.__class__.from_dict(c.to_dict()); s.vocab_size=1000; s.num_hidden_layers=2; s.layer_types=['linear_attention','full_attention']
    if name.endswith('27B'): s.hidden_size=1024; s.intermediate_size=3584
    z=AutoModelForCausalLM.from_config(s).eval()
    ids=torch.randint(0,1000,(1,8))
    with torch.no_grad():
        for p in z.parameters(): p.zero_()
        o=z(input_ids=ids).logits
    print(name,type(z).__name__,'zero: finite',bool(o.isfinite().all()),'max',float(o.nan_to_num().abs().max()))
    print('  nan?',bool(o.isnan().any()),'tensor kinds',sorted({n.split('.')[-2] for n,_ in z.named_parameters()})[:30])
