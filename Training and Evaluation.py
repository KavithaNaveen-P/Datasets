device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GACL_ECGNet(input_dim=256, hidden_dim=128, output_dim=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Hybrid loss
def hybrid_loss(logits, y, embeddings, model, lambda_=0.6):
    ce_loss = F.cross_entropy(logits, y)
    contrastive_loss = model.adaptive_contrastive_loss(embeddings, y)
    return lambda_ * ce_loss + (1 - lambda_) * contrastive_loss

# Training loop
def train(model, data):
    model.train()
    optimizer.zero_grad()
    logits, embeddings = model(data)
    loss = hybrid_loss(logits, data.y, embeddings, model)
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluation
def test(model, data):
    model.eval()
    with torch.no_grad():
        logits, _ = model(data)
        preds = logits.argmax(dim=1)
        acc = (preds == data.y).sum().item() / data.y.size(0)
    return acc

# Train for 100 epochs
for epoch in range(100):
    loss = train(model, graph_data.to(device))
    acc = test(model, graph_data.to(device))
    print(f'Epoch {epoch+1:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}')
