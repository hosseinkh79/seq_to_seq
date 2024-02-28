import torch
import torch.nn as nn
import torchtext

import datasets
import spacy

dataset = datasets.load_dataset('bentrevett/multi30k')

train_data, valid_data, test_data = (dataset['train'], dataset['validation'], dataset['test'])

en_nlp = spacy.load('en_core_web_sm')
de_nlp = spacy.load('de_core_news_sm')


def tokenize_example(example, en_nlp, de_nlp, max_length, lower, sos_token, eos_token):
    en_tokens = [token.text for token in en_nlp.tokenizer(example['en'])][:max_length]
    de_tokens = [token.text for token in de_nlp.tokenizer(example['de'])][:max_length]
    if lower: 
        en_tokens = [token.lower() for token in en_tokens]
        de_tokens = [token.lower() for token in de_tokens]
    en_tokens = [sos_token] + en_tokens + [eos_token]
    de_tokens = [sos_token] + de_tokens + [eos_token]

    return {'en_tokens': en_tokens, 'de_tokens': de_tokens}



max_length = 1_000
lower = True
sos_token = "<sos>"
eos_token = "<eos>" 

fn_kwargs = {
    'en_nlp': en_nlp,
    'de_nlp': de_nlp,
    'max_length': max_length,
    'lower': lower,
    'sos_token': sos_token,
    'eos_token': eos_token
}


train_data = train_data.map(function=tokenize_example, fn_kwargs=fn_kwargs)
valid_data = valid_data.map(function=tokenize_example, fn_kwargs=fn_kwargs)
test_data = test_data.map(function=tokenize_example, fn_kwargs=fn_kwargs)





min_freq = 2
unk_token = '<unk>'
pad_token = '<pad>'

special_tokens = [
    unk_token,
    pad_token,
    sos_token,
    eos_token,
]

en_vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data['en_tokens'],
    min_freq=min_freq,
    specials=special_tokens
)

de_vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data['de_tokens'],
    min_freq=min_freq,
    specials=special_tokens
)



# assert en_vocab[unk_token] == de_vocab[unk_token]
# assert en_vocab[pad_token] == de_vocab[pad_token]

unk_index = en_vocab[unk_token]
pad_index = en_vocab[pad_token]
en_vocab.set_default_index(unk_index)
de_vocab.set_default_index(unk_index)



def numericalize_example(example, en_vocab, de_vocab):
    en_ids = en_vocab.lookup_indices(example['en_tokens'])
    de_ids = de_vocab.lookup_indices(example['de_tokens'])

    return {'en_ids': en_ids, 'de_ids': de_ids}

fn_kwargs = {"en_vocab": en_vocab, "de_vocab": de_vocab}

train_data = train_data.map(numericalize_example, fn_kwargs=fn_kwargs)
valid_data = valid_data.map(numericalize_example, fn_kwargs=fn_kwargs)
test_data = test_data.map(numericalize_example, fn_kwargs=fn_kwargs)



data_type = "torch"
format_columns = ["en_ids", "de_ids"]

train_data = train_data.with_format(
    type=data_type, columns=format_columns, output_all_columns=True
)

valid_data = valid_data.with_format(
    type=data_type,
    columns=format_columns,
    output_all_columns=True,
)

test_data = test_data.with_format(
    type=data_type,
    columns=format_columns,
    output_all_columns=True,
)



def collate_fn(batch):
    en_ids = [example['en_ids'] for example in batch]
    de_ids = [example['de_ids'] for example in batch]
    padded_en = nn.utils.rnn.pad_sequence(sequences=en_ids, padding_value=pad_index)
    padded_de = nn.utils.rnn.pad_sequence(sequences=de_ids, padding_value=pad_index)

    batch =  {
        'en_ids': padded_en, 
        'de_ids': padded_de
        }

    return batch



batch_size = 32
train_data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_data_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_data_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

def get_loaders():
    return train_data_loader, valid_data_loader, test_data_loader




def translate_sentence( sentence,
                        model,
                        device,
                        max_output_length=25,
                        de_nlp=de_nlp,
                        en_vocab=en_vocab,
                        de_vocab=de_vocab,
                        lower=True,
                        sos_token=sos_token,
                        eos_token=eos_token):
    print(f'1')
    model = model.to(device)
    print(f'11')
    model.eval()
    print(f'111')

    with torch.no_grad():
        print(f'1111')
        if isinstance(sentence, str):
            tokens = [token.text for token in de_nlp.tokenizer(sentence)]
        else:
            tokens = [token for token in sentence]
        if lower:
            print(f'11111')
            tokens = [token.lower() for token in tokens]
        print(f'111111')
        tokens = [sos_token] + tokens + [eos_token]
        ids = de_vocab.lookup_indices(tokens)
        tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)
        print(f'2')
        hidden, cell = model.encoder(tensor)
        print(f'22')
        inputs = en_vocab.lookup_indices([sos_token])
        print(f'222')
        for _ in range(max_output_length):
            print(f'3')
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            print(f'33')
            output, hidden, cell = model.decoder(inputs_tensor, hidden, cell)
            print(f'333')
            predicted_token = output.argmax(-1).item()
            print(f'4')
            inputs.append(predicted_token)
            print(f'44')
            if predicted_token == en_vocab[eos_token]:
                print(f'444')
                break
        tokens = en_vocab.lookup_tokens(inputs)
    return tokens




