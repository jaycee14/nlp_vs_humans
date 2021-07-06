"""Util functions"""

def get_config(filepath):
    config = {}
    with open(filepath, 'r') as f:
        for line in f:
            k, v = line.rstrip().split('=')
            config[k] = v

    return config


def create_db_url(config, host_addr):
    url = 'postgresql+psycopg2://{user}:{pw}@{url}/{db}'.format(user=config['POSTGRES_USER'],
                                                                pw=config['POSTGRES_PASSWORD'],
                                                                url=host_addr,
                                                                db=config['POSTGRES_DB'], )

    return url


def reset_model(path):
    """ loads model and prepares layers"""
    
    bs = 64
    data_clas = load_data(path, 'data_clas_32.pkl', bs=bs)
    learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
    learn.load_encoder('fine_tuned_enc_32')
    learn.load('third_32')

    model = learn.model.cpu()
    model[0].bptt = 500
    model.eval()

    model.zero_grad()
    embedding = model[0].module.encoder_dp

    return model, embedding, data_clas


def prepare_captum_tensors(input_tensor):
    """version without chunking - just prepares a padding tensor"""

    pad_tensor = torch.ones_like(input_tensor).long().cpu()
    pad_tensor[0, 0] = 2

    return pad_tensor


def explain_tweet(sentiment, input_tensor, lig,n_steps=25):
    """Run captum process on a tweet"""

    padding_tensor = prepare_captum_tensors(input_tensor)

    target = 1 if sentiment == 'pos' else 0

    attributions_ig, delta = lig.attribute(input_tensor, padding_tensor, target=target, n_steps=n_steps,
                                           return_convergence_delta=True)

    attributions_ig = attributions_ig.sum(dim=2).squeeze(0)

    attributions = attributions_ig / torch.norm(attributions_ig)
    attributions = attributions.detach().numpy()

    return attributions


class IMDBClassifier(nn.Module):
    
    def __init__(self, n_classes, base_model):
        
        super(IMDBClassifier, self).__init__()
        self.model = base_model
        self.drop=nn.Dropout(p=0.3)
        self.out = nn.Linear(self.model.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        drop_out = self.drop(output[0][:, 0, :])
        return self.out(drop_out)


class IMDBDataset(Dataset):
    
    def __init__(self, texts, labels, found_labels, tokenizer, max_len=512 ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.found_labels = found_labels
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        
        text = str(self.texts[item])
        label = self.labels[item]
        
        found_label = self.found_labels[item][1:-1].split(',')
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {'text':text,
               'input_ids':encoding['input_ids'].flatten(),
               'attention_mask':encoding['attention_mask'].flatten(),
               'label':torch.tensor(label,dtype=torch.long),
               'found_label':found_label}

    
def create_data_loader(texts,labels, found_labels, tokenizer, max_len, batch_size):

    ds = IMDBDataset(
        texts=texts,
        labels=labels,
        found_labels=found_labels,
        tokenizer=tokenizer,
        max_len=max_len)

    return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=4)


def explain_tweet_bert(batch_element, lig,n_steps=50):

    padding_tensor = torch.zeros_like(batch_element['input_ids']).long().cpu()


    attributions_ig, delta = lig.attribute(batch_element['input_ids'], 
                                            padding_tensor, 
                                            target=batch_element['label'].long().cpu(), 
                                            n_steps=n_steps,
                                            return_convergence_delta=True,
                                            additional_forward_args=batch_element['attention_mask'])

    attributions_ig = attributions_ig.sum(dim=2).squeeze(0)

    attributions = attributions_ig / torch.norm(attributions_ig)
    attributions = attributions.detach().numpy()


    return attributions