import torch


def stack_batch_time(x):
    # [B, C, T, ...]
    x = x.movedim(2, 1)  # [B, T, C, ...]
    x = x.flatten(0, 1)  # [B * T, C, ...]
    return x


def unstack_batch_time(x, batch_size):
    # [B * T, C, ...]
    x = x.unflatten(0, (batch_size, -1))  # [B, T, C, ...]
    x = x.movedim(1, 2)  # [B, C, T, ...]
    return x


def forward(x, batch_size, encoder, ar_model):
    x = stack_batch_time(x)
    latents = encoder(x)
    latents = unstack_batch_time(latents, batch_size=batch_size)
    context = ar_model(latents)
    return context, latents


def cal_density_ratio(context, latents, n_time, batch_size, query_weights):
    latents = torch.movedim(latents, 0, 1)
    latents = torch.flatten(latents, 1, 2)
    latents = latents[:, :, None]
    loss = None
    count = 0
    for i in range(n_time - 1):
        for k in range(1, n_time - i):
            query = context[:, :, [i]]
            query = stack_batch_time(query)
            query = query_weights[k - 1](query)
            query = unstack_batch_time(query, batch_size)
            query = torch.movedim(query, 0, 1)
            query = torch.flatten(query, 1, 2)
            query = query[:, None]
            dot = torch.mul(query, latents).sum(dim=0)
            density_ratio = torch.nn.functional.log_softmax(dot, dim=0)
            pos_pairs = [
                density_ratio[i + k + n_time * j, j] for j in range(batch_size)
            ]
            pos_pairs = torch.stack(pos_pairs, dim=-1)
            count += 1
            if loss is None:
                loss = pos_pairs
            else:
                loss = loss + pos_pairs
    return -loss.sum() / (batch_size * count)


def cal_loss(x, batch_size, n_time, encoder, ar_model, query_weights):
    context, latents = forward(
        x=x, batch_size=batch_size, encoder=encoder, ar_model=ar_model
    )
    loss = cal_density_ratio(
        context=context,
        latents=latents,
        n_time=n_time,
        batch_size=batch_size,
        query_weights=query_weights,
    )
    return loss