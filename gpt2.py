### importar pacotes ###
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# O GPT2Tokenizer vai pegar o texto de entrada e codificá-lo na forma de
# números. Os números serão passados para o modelo GPT-2. No final o 
# GPT2Tokenizer fará a operação inversa, irá transformar os números de
# saída do modelo em texto novamente.

### Carregar o modelo ###
# carrega o tokenizer com o modelo pré-treinado
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

# carrega o modelo pré-treinado
# pad_token_id representa qual token será usado para preencher o texto
# tokenizer.eos_token_id representa o token de final de frase.
model = GPT2LMHeadModel.from_pretrained('gpt2-large', pad_token_id=tokenizer.eos_token_id)

### Tokenizar uma sentença ###
# criar uma string (sentença) para entrada no modelo
sentence = "explain to me what is thermodynamics"
# tokenizar -> codificar a frase em números que identificam as palavras
# da sentença. No final o retorno de tensores que é do tipo tensor pytorch
input_ids = tokenizer.encode(sentence, return_tensors='pt')

### Gerar o texto e decodificá-lo ###
# max_length -> quantidade máxima de tokens no texto gerado somado com 
# os tokens do prompt.

# max_new_tokens -> quantidade máxima de tokens no texto gerado ignorando
# os tokens do prompt.

# num_beams -> método beams search para encontrar a próxima palavra mais
# adequada na sequência. O número de beams determina quantas árvores de
# pesquisa será utilizada.

# no_repeat_ngram_size -> impede que o modelo repita certas sequências.
# Com o número 2 a sequência só pode ser repetida uma vez.

# early_stopping -> se não estiverem melhoram os resultados, o modelo
# para de gerar o resultado.
output = model.generate(input_ids, max_new_tokens=500, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

### Resultado ###
# decode para decodificar a saída. Números -> Texto
# output[0] -> pega a primeira parte da matriz
# skip_special_tokens = True -> para não exibir tokens no final de frase,
# nem tokens especiais.
text = tokenizer.decode(output[0], skip_special_tokens=True)
print(text)
