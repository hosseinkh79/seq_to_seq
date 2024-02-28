from going_modular.configs import configs
import torch
from tqdm import tqdm

def one_step_train(model,
                   data_loader,
                   loss_fn,
                   optimizer,
                   device):
    
    model = model.to(device)
    model.train()
    
    train_loss = 0
    for i, batch in enumerate(data_loader):
        optimizer.zero_grad()

        src = batch['de_ids'].to(device)
        trg = batch['en_ids'].to(device)

        output = model(src, trg, configs['teacher_force_ratio'])
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)

        trg = trg[1:].view(-1)

        loss = loss_fn(output, trg)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), configs['clip'])
        optimizer.step()

        train_loss += loss.item()

    total_loss = train_loss/len(data_loader)

    return total_loss

def one_step_test(model,
                   data_loader,
                   loss_fn,
                   optimizer,
                   device):
    
    model = model.to(device)
    model.eval()
    
    test_loss = 0
    with torch.inference_mode():
        for i, batch in enumerate(data_loader):

            src = batch['de_ids'].to(device)
            trg = batch['en_ids'].to(device)

            output = model(src, trg, 0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)

            trg = trg[1:].view(-1)

            loss = loss_fn(output, trg)

            test_loss += loss.item()

        total_loss = test_loss/len(data_loader)

    return total_loss


def train(model,
          train_loader,
          test_loader,
          loss_fn,
          optimizer,
          device,
          epochs):
    
    results = {
        'train_loss': [],
        'test_loss': []
    }

    for epoch in range(epochs):

        train_loss = one_step_train(model=model,
                                    data_loader=train_loader,
                                    loss_fn=loss_fn,
                                    optimizer=optimizer,
                                    device=device)
        
        test_loss = one_step_test(model=model,
                                    data_loader=test_loader,
                                    loss_fn=loss_fn,
                                    optimizer=optimizer,
                                    device=device)
        
        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)

        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"test_loss: {test_loss:.4f} | "
        )
        
    return results

