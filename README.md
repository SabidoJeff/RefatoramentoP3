# RefatoramentoP3
Esse repositório trata-se do projeto de refatoramento da matéria projetos de software. O código utilizado para exemplificar as correções de bad smells é uma parte de um projeto de processamento de texto devenvolvida por mim.

No objeto ConsecutiveNPChunker o construtor estava relizando a função de marcar as frases do texto, o que vai além do que é suposto realizar, considerado um long method, desse modo, com o padrão Extract Method, foi criado método exclusivo para o processamento dos textos.

