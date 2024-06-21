import torch
#from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from accelerate import Accelerator
import torch.nn.functional as F

# 使用 Accelerator 初始化 FSDP
#accelerator = Accelerator(fp16=True, split_batches=True)
accelerator = Accelerator()
device = accelerator.device
#model = accelerator.prepare_model(model)

# 加载预训练模型和分词器
#tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
#model = AutoModel.from_pretrained('bert-base-uncased').to(device)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased').to(device)



# 准备数据和训练逻辑
def train(model, tokenizer, accelerator):
    # 假设有一些训练数据和目标
    inputs = tokenizer(["Hello, this is a test."], return_tensors="pt", padding=True, truncation=True)['input_ids'].to(accelerator.device)
    labels = torch.LongTensor([1]).to(accelerator.device)  # 假设目标
    data = (inputs, labels)

    optimizer = torch.optim.Adam(model.parameters())

    model, optimizer, data = accelerator.prepare(model, optimizer, data)

    # 将数据移动到相应的设备
    #inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
    #labels = labels.to(accelerator.device)

    # 训练模式
    model.train()
    for epoch in range(10):
        print(f"training epoch {epoch}")
        inputs, labels = data
        outputs = model(inputs).logits
        loss = F.cross_entropy(outputs, labels)

        #loss.backward()
        accelerator.backward(loss)

        # 更新模型参数
        optimizer.step()
        optimizer.zero_grad()

# 训练模型
train(model, tokenizer, accelerator)

