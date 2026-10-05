# Custom 300M Param GPT LLM
## At a glance:
This repo contains a custom made ~300 million parameter GPT-style language model, 
with the tokenizer, architecure, and training stack written 
entirely from scratch. The model was trained on roughly 14.75 Billion tokens,
tokens of OpenWebText2, on two RTX 3090s for nearly 100 hours. totalling nearly 30 ExaFLOP of compute. Final loss was 
around 3.1

## Results at a glance:

![Loss](figures/loss_curve.png)

| Parameters               | 300,648,448                               |
|:-------------------------|:------------------------------------------|
| Context / Vocab          | 1,024 / 127k  <br/> byte-level BPE        |
| Training Data            | OpenWebText2, 14.75B tokens, 1 epoch      |
| Hardware                 | 2x RTX 3090                               |
| Wall-clock               | TODO                                      |
| Final Training Loss      | 3.11                                      |
| Loss on slice of dataset | 3.03                                      |
| Hellaswag, 0-shot        | acc_norm 0.325, acc 0.297 (random = 0.25) |

### Sample generations
- Prompt: 1+1=
  - **Response:**
        
        1+1=100000*1000*1000000
        
        0 0 0 0 0 0 0 0 0 0 0
        
        The size of the buffer is defined as the number of bytes in the memory address.
        
        We could have written:
        
        mov eax, d⭐ ptr [esi] ; mov edi, d⭐ ptr [esi+ecx], esi;
        
        However, this will result in an infinite loop, because the value of the second argument (esp) will be equal to
- Prompt: The history of the
  - **Response:**
        
        The history of the US and the UK’s relationship with the EU is a complex one, but it has been a long-running one.

        It began in the early 1970s when the UK was granted membership of the European Economic Community (EEC) after the Second World War. The UK joined the EEC in 1973upt the terms of the Treaty of Rome, which established the bloc as a single market for goods and services.

        In 1979, the UK joined the European Economic Community (EEC) as an observer state, and