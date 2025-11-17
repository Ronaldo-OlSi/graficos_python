import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("ecommerce_estatistica.csv")

print(df.head())
print(df.info())
print(df.describe(include="all"))

# HISTOGRAMA – DISTRIBUIÇÃO DE NOTA (Qualidade do produto)
plt.figure(figsize=(8,5))
plt.hist(df["Nota"].dropna(), bins=20)
plt.title("Distribuição das Notas dos Produtos")
plt.xlabel("Nota")
plt.ylabel("Frequência")
plt.grid(alpha=0.3)
plt.show()

# DISPERSÃO – NOTA vs NÚMERO DE AVALIAÇÕES. (Identificar se produtos mais avaliados têm melhores notas)
plt.figure(figsize=(8,6))
plt.scatter(df["N_Avaliações"], df["Nota"], alpha=0.6)
plt.title("Dispersão: N° Avaliações vs Nota")
plt.xlabel("Nº de Avaliações")
plt.ylabel("Nota")
plt.grid(alpha=0.3)
plt.show()

# MAPA DE CALOR – CORRELAÇÃO ENTRE CAMPOS NUMÉRICOS. (Descobrir relações entre métricas importantes)
num_cols = df.select_dtypes(include=[np.number])
corr = num_cols.corr()

plt.figure(figsize=(10,6))
plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
plt.colorbar(label="Correlação")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Mapa de Calor das Correlações entre Variáveis Numéricas")
plt.tight_layout()
plt.show()

# GRÁFICO DE BARRA – VENDAS POR MARCA. (Entender quais marcas mais vendem)

# Converter Qtd_Vendidos para numérico
df["Qtd_Vendidos"] = pd.to_numeric(df["Qtd_Vendidos"], errors="coerce")

vendas_por_marca = df.groupby("Marca")["Qtd_Vendidos"].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
plt.bar(vendas_por_marca.index, vendas_por_marca.values)
plt.title("Top 10 Marcas por Quantidade de Produtos Vendidos")
plt.xlabel("Marca")
plt.ylabel("Quantidade Vendida")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# GRÁFICO DE PIZZA – DISTRIBUIÇÃO DE GÊNERO DOS PRODUTOS. (Quais públicos predominam na loja?)

# Contagem completa dos gêneros
generos = df["Gênero"].value_counts()

# Separa os 3 gêneros mais frequentes
top3 = generos.head(3)

# Soma dos demais gêneros
outros_total = generos.iloc[3:].sum()

# Cria nova série com Top3 + Outros
generos_agrupados = top3.copy()
generos_agrupados["Outros"] = outros_total

# Gráfico de pizza com os 4 grupos finais
plt.figure(figsize=(7,7))
plt.pie(
    generos_agrupados.values,
    labels=generos_agrupados.index.astype(str),
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Distribuição de Gênero dos Produtos - Quais públicos predominam?")
plt.axis("equal")
plt.show()

# DENSIDADE – PREÇOS (APÓS DESCONTO). (Analisar a concentração de preços vendidos)

# Converter o campo Desconto para número (caso venha como texto)
df["Desconto"] = pd.to_numeric(df["Desconto"], errors="coerce")
plt.figure(figsize=(8,5))

# Plotar KDE
ax = df["Desconto"].dropna().plot(kind="kde")

plt.title("Densidade do Valor dos Descontos")
plt.xlabel("Desconto (R$)")
plt.ylabel("Densidade")
plt.grid(alpha=0.3)
# --- formatar o eixo X para 1 casa decimal ---
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))
plt.show()

# GRÁFICO DE REGRESSÃO — Nota vs Nº de Avaliações
x = df["Preço_MinMax"]
y = df["Desconto_MinMax"]

# Remover valores nulos ou inválidos se existirem
mask = x.notna() & y.notna()
x = x[mask]
y = y[mask]

# Evitar erro caso não existam dados suficientes
if len(x) < 2:
    print("Não há dados suficientes para gerar a regressão.")
else:
    plt.figure(figsize=(8,6))
    plt.scatter(x, y, alpha=0.5, label="Dados reais")
    coef = np.polyfit(x, y, 1)
    pol = np.poly1d(coef)

    xs = np.linspace(x.min(), x.max(), 100)
    plt.plot(xs, pol(xs), color="red", linewidth=2,
             label=f"Linha de Regressão: y = {coef[0]:.3f}x + {coef[1]:.3f}")

    # Personalização do gráfico
    plt.title("Regressão: Preço MinMax vs Desconto MinMax\nProdutos caros recebem mais descontos?")
    plt.xlabel("Preço Normalizado (Preço_MinMax)")
    plt.ylabel("Desconto Normalizado (Desconto_MinMax)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()