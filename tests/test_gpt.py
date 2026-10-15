import torch

from nanochat.gpt import GPT, GPTConfig, has_ve


def make_tiny_gpt():
    config = GPTConfig(
        sequence_len=8,
        vocab_size=128,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=32,
        window_pattern="L",
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()
    return model


def test_value_embedding_is_one_shared_table():
    model = make_tiny_gpt()

    value_embedding_params = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "value_embed" in name
    ]

    assert [name for name, _ in value_embedding_params] == ["value_embed.weight"]
    assert all(parameter.shape == (128, 32) for _, parameter in value_embedding_params)
    assert model.num_scaling_params()["value_embeds"] == 128 * 32


def test_shared_value_embedding_forward_and_backward():
    model = make_tiny_gpt()
    selected_layers = sum(has_ve(i, model.config.n_layer) for i in range(model.config.n_layer))
    lookup_calls = 0

    def count_lookup(_module, _inputs, _output):
        nonlocal lookup_calls
        lookup_calls += 1

    handle = model.value_embed.register_forward_hook(count_lookup)
    idx = torch.randint(0, model.config.vocab_size, (2, 8))
    targets = torch.randint(0, model.config.vocab_size, (2, 8))
    loss = model(idx, targets)
    loss.backward()
    handle.remove()

    assert selected_layers == 2
    assert lookup_calls == 1
    assert torch.isfinite(loss)
    assert model.value_embed.weight.grad is not None
    assert torch.isfinite(model.value_embed.weight.grad).all()


def test_chunked_logits_match_full_loss_and_gradients():
    model = make_tiny_gpt()
    idx = torch.randint(0, model.config.vocab_size, (2, 8))
    targets = torch.randint(0, model.config.vocab_size, (2, 8))
    targets[0, :2] = -1

    model.config.logit_chunk_size = 0
    full_losses = model(idx, targets, loss_reduction="none").detach()
    full_loss = model(idx, targets)
    full_loss.backward()
    full_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }

    model.zero_grad(set_to_none=True)
    model.config.logit_chunk_size = 5
    chunked_losses = model(idx, targets, loss_reduction="none").detach()
    chunked_loss = model(idx, targets)
    chunked_loss.backward()

    torch.testing.assert_close(chunked_losses, full_losses, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(chunked_loss, full_loss, rtol=1e-6, atol=1e-6)
    for name, parameter in model.named_parameters():
        if name in full_grads:
            torch.testing.assert_close(parameter.grad, full_grads[name], rtol=1e-5, atol=1e-6)
