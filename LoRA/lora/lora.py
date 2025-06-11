import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer


class LoRALinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        lora_alpha: int,
        bias: bool = True,
        merge_weights: bool = False,
    ):
        super().__init__(in_features, out_features, bias=bias)

        self.r = r
        self.lora_alpha = lora_alpha
        self.merge_weights = merge_weights

        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            self.scaling = self.lora_alpha / r

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x) -> torch.Tensor:
        if self.r > 0 and not self.merge_weights:
            lora_out = F.linear(x, self.weight, self.bias)
            lora_out += (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return lora_out
        else:
            return F.linear(x, self.weight, self.bias)

    def merge(self):
        if self.r > 0 and not self.merge_weights:
            delta_w = self.lora_B @ self.lora_A
            self.weight.data += delta_w * self.scaling
            self.merge_weights = True

    def unmerge(self):
        if self.r > 0 and self.merge_weights:
            delta_w = self.lora_B @ self.lora_A
            self.weight.data -= delta_w * self.scaling
            self.merge_weights = False


def inject_lora(model: nn.Module, target_modules: list, r: int, lora_alpha: int) -> nn.Module:
    for param in model.parameters():
        param.requires_grad = False  # Freeze all parameters

    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name in target_modules:
            lora_module = LoRALinear(
                in_features=module.in_features,
                out_features=module.out_features,
                r=r,
                lora_alpha=lora_alpha,
                bias=module.bias is not None,
            )
            lora_module.weight.data = module.weight.data.clone()
            if module.bias is not None:
                lora_module.bias.data = module.bias.data.clone()
            setattr(model, name, lora_module)
        else:
            inject_lora(module, target_modules, r, lora_alpha)
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
    return model


if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3-8B"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Injecting LoRA into the model...")
    inject_lora(model, ["q_proj", "k_proj", "v_proj", "o_proj"], r=4, lora_alpha=4)

    print("Trainable parameters:")
    print([n for n, p in model.named_parameters() if p.requires_grad])


    def generate(prompt, max_new_tokens=20):
        ids = tokenizer(prompt, return_tensors="pt")
        gen = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1,
            top_p=1,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(gen[0], skip_special_tokens=True)


    print("Before fine-tune:")
    print(generate("Hello Wayne's Talk"))

    print("Fine-tuning the model...")
    model.train()
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    train_text = "Wayne's Talk is a technical blog about mobile, frontend, backend and AI."
    inputs = tokenizer(train_text, return_tensors="pt")
    for step in range(10):
        outputs = model(**inputs, labels=inputs["input_ids"])
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    print("After fine-tune (unmerged):")
    print(generate("Hello Wayne's Talk"))

    print("Merging LoRA weights...")
    merge_lora(model)

    print("After fine-tune (merged):")
    print(generate("Hello Wayne's Talk"))
