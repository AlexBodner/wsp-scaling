import logging; logging.disable(logging.WARNING)
from transformers import AutoTokenizer
T={'gpt2':AutoTokenizer.from_pretrained('gpt2'),'qwen3.5':AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B')}
samples=['12+34=',' 46','rev: a b c',' c b a','The capital of France is',' Paris',' Ouagadougou',' Reykjavik',' Tokyo','Wall St. Bears Claw Back Into the Black\ntopic:',' sports',' business',' world',' science','Once upon a time there was a little girl']
for n,t in T.items():
    print(n,'special BOS?',t.bos_token,'eos',t.eos_token)
    for s in samples:
        ids=t(s,add_special_tokens=False).input_ids
        print('  ',repr(s),len(ids),[t.decode([i]) for i in ids][:12])
    print('   digits one token each:',all(len(t(str(d),add_special_tokens=False).input_ids)==1 for d in range(10)),'; two-digit one token:',sum(len(t(str(d),add_special_tokens=False).input_ids)==1 for d in range(10,100)),'/90')
    print('   spaced letters one token:',sum(len(t(' '+c,add_special_tokens=False).input_ids)==1 for c in 'abcdefghijklmnopqrstuvwxyz'),'/26')
