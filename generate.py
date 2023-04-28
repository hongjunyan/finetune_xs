from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers
from packaging import version


assert version.parse(transformers.__version__) >= version.parse("4.23.0")



checkpoint = "./xs-poly-160M/checkpoint-5000/"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)


# ------ (Not support neox)GPU Parallel setting ------
# from parallelformers import parallelize
# parallelize(model, num_gpus=2, fp16=True, verbose='detail', custom_policies=gpt_neo.GPTNeoPolicy)


# ------ Method1 - Directly use model to get prediction ------
inputs = tokenizer("用XS寫RSI黃金交叉和爆量超過1000張\n\n###\n\n", return_tensors="pt")

outputs = model.generate(
    **inputs,
    num_beams=2,
    # no_repeat_ngram_size=4,
    max_length=512,
)


res = tokenizer.batch_decode(outputs)[0]
res = res.replace("\n\n###\n\n", "\n\n")
print(res.split("###")[0].strip())


# ------ Method2 - Use Pipeline to get prediction ------
import torch
from transformers import pipeline

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, device=device
)

prompt = """用XS寫RSI黃金交叉和爆量超過1000張\n\n###\n\n"""
res = pipe(prompt, num_return_sequences=1, max_length=512, return_full_text=False)

for i in res:
    #print(i["generated_text"])
    output = i["generated_text"].replace("\n\n###\n\n", "\n\n")
    print(output.split("###")[0].strip())
    print("-"*60)