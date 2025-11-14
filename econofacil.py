import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.ar_model import AutoReg
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="EconoFácil - Descomplicando a Grana",
    page_icon="📊",
    layout="wide"
)

# Título e branding
st.title("📊 **EconoFácil**")
st.markdown("### *Econometria Profissional para Todos*")
st.markdown("---")

# Sidebar com informações
with st.sidebar:
    st.header("🔧 **Como Funciona**")
    st.write("""
    1. **Upload**: Carregue seus dados em Excel/CSV
    2. **Análise**: Modelo econométrico automático
    3. **Resultados**: Gráficos e projeções profissionais
    4. **Download**: Relatório completo
    """)

    st.header("💰 **Planos**")
    st.write("**Free**: Análise básica")
    st.write("**Pro**: R$29/mês - Relatórios completos")
    st.write("**Business**: R$299/mês - Consultoria")

    st.header("💬 **Entre em Contato**")
    if st.button("📱 WhatsApp Business"):
        st.markdown("[Fale comigo no WhatsApp](https://wa.me/5511967273149?text=Olá!%20Testei%20o%20EconoFácil%20e%20gostei%20muito!)", unsafe_allow_html=True)

    st.info("Desenvolvido por Cristiane Graziela")

# ============================================================================
# PASSO 1: UPLOAD DE DADOS
# ============================================================================
st.header("📁 **Passo 1: Carregue seus Dados**")

col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Escolha um arquivo Excel ou CSV",
        type=['xlsx', 'csv'],
        help="Formato: colunas 'Ano', 'Consumo', 'Juros', 'Inflacao'"
    )

with col2:
    st.info("💡 Use dados de exemplo para testar!")

# Dados de exemplo
if st.button("🔄 **Usar Dados de Exemplo**"):
    df_exemplo = pd.DataFrame({
        'Ano': [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
        't': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        'Consumo': [390, 375, 368, 355, 342, 342, 338, 342, 348, 352, 362, 383, 399, 402, 405, 410, 415],
        'Juros': [10, 11, 12, 13, 14, 14, 14, 13, 14, 14, 14, 14, 13, 15, 13.75, 10.50, 9.00],
        'Inflacao': [87, 86, 85, 82, 79, 78, 78, 75, 75, 75, 76, 72, 76, 79, 4.62, 3.80, 3.50]
    })
    st.session_state.df = df_exemplo
    st.success("✅ Dados de exemplo carregados!")

# Carregar dados do usuário
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        # Detectar colunas automaticamente
        colunas_possiveis = {
            'Ano': ['ano', 'year', 'data'],
            'Consumo': ['consumo', 'y', 'consumo_familiar'],
            'Juros': ['juros', 'selic', 'i', 'taxa'],
            'Inflacao': ['inflacao', 'ipca', 'pi', 'inflação']
        }

        mapeamento = {}
        for col_padrao, variacoes in colunas_possiveis.items():
            for col_real in df.columns:
                if any(var in col_real.lower() for var in variacoes):
                    mapeamento[col_real] = col_padrao
                    break

        if len(mapeamento) < 4:
            st.warning("⚠️  Selecione as colunas manualmente:")
            ano_col = st.selectbox("Coluna do Ano:", df.columns)
            cons_col = st.selectbox("Coluna do Consumo:", df.columns)
            juros_col = st.selectbox("Coluna dos Juros:", df.columns)
            infl_col = st.selectbox("Coluna da Inflação:", df.columns)

            df_padronizado = pd.DataFrame({
                'Ano': df[ano_col],
                't': range(1, len(df) + 1),
                'Consumo': df[cons_col],
                'Juros': df[juros_col],
                'Inflacao': df[infl_col]
            })
        else:
            df_padronizado = df.rename(columns=mapeamento)
            df_padronizado['t'] = range(1, len(df_padronizado) + 1)

        st.session_state.df = df_padronizado
        st.success(f"✅ Dados carregados: {len(df_padronizado)} observações")
        st.dataframe(df_padronizado.head())

    except Exception as e:
        st.error(f"❌ Erro ao carregar: {str(e)}")

# ============================================================================
# PASSO 2: ANÁLISE AUTOMÁTICA
# ============================================================================
if 'df' in st.session_state:
    st.header("🔬 **Passo 2: Análise Econométrica Automática**")

    df = st.session_state.df.copy()

    # Preparar dados
    df['Juros_decimal'] = df['Juros'] / 100
    df['Inflacao_decimal'] = df['Inflacao'] / 100
    df['ln_Consumo'] = np.log(df['Consumo'])

    # Modelo
    X = df[['Juros_decimal', 'Inflacao_decimal', 't']].copy()
    X = sm.add_constant(X)
    y = df['ln_Consumo'].copy()

    try:
        # Modelo tradicional
        modelo_trad = sm.OLS(y, X).fit()
        dw_trad = durbin_watson(modelo_trad.resid)

        # Correção GLS
        residuos = modelo_trad.resid.values
        ar1_model = AutoReg(residuos, lags=1, old_names=False).fit()
        rho = ar1_model.params[1] if len(ar1_model.params) > 1 else 0.5

        y_gls = y - rho * y.shift(1)
        X_gls = X - rho * X.shift(1)
        y_gls = y_gls.iloc[1:]
        X_gls = X_gls.iloc[1:]

        from statsmodels.regression.linear_model import GLS
        modelo_final = GLS(y_gls, X_gls).fit()
        dw_final = durbin_watson(modelo_final.resid)

        st.success(f"✅ **Análise Concluída!**")
        st.success(f"• R² Ajustado: **{modelo_final.rsquared_adj:.3f}** ({modelo_final.rsquared_adj*100:.1f}%)")
        st.success(f"• Durbin-Watson: **{dw_final:.3f}**")

        # Equação
        st.subheader("📐 **Equação Estimada**")
        st.latex(f"\\ln(Consumo) = {modelo_final.params[0]:.3f} {modelo_final.params[1]:+.3f} \\cdot Juros {modelo_final.params[2]:+.3f} \\cdot Inflação {modelo_final.params[3]:+.3f} \\cdot Tempo")

        st.markdown(f"""
        **Interpretação:**
        - Cada 1 p.p. ↑ nos **juros** → Consumo **{modelo_final.params[1]*100:+.2f}%**
        - Cada 1 p.p. ↑ na **inflação** → Consumo **{modelo_final.params[2]*100:+.2f}%**
        - **Tendência anual**: **+{modelo_final.params[3]*100:.2f}%** no consumo
        """)

        # Gráficos
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 **Real vs Previsto**")
            df['Pred_ln'] = modelo_trad.predict(X)
            df['Pred_Consumo'] = np.exp(df['Pred_ln'])

            fig1 = px.line(df, x='Ano', y=['Consumo', 'Pred_Consumo'], 
                          title="Ajuste do Modelo")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.subheader("📊 **Resíduos**")
            fig2 = px.scatter(df, x='Ano', y=modelo_trad.resid, 
                             title=f"DW = {dw_final:.3f}")
            fig2.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig2, use_container_width=True)

        # ============================================================================
        # PASSO 3: PROJEÇÕES
        # ============================================================================
        st.header("🔮 **Passo 3: Projeções 2026**")

        cenarios = {
            'Base': {'Juros': 8.5, 'Inflacao': 3.25},
            'Otimista': {'Juros': 7.5, 'Inflacao': 2.5},
            'Pessimista': {'Juros': 10.5, 'Inflacao': 4.5}
        }

        t_futuro = len(df) + 1
        projecoes = []

        col1, col2, col3 = st.columns(3)

        for i, (nome, params) in enumerate(cenarios.items()):
            with [col1, col2, col3][i]:
                st.subheader(f"**{nome.upper()}**")

                X_futuro = pd.DataFrame({
                    'const': [1], 
                    'Juros_decimal': [params['Juros']/100], 
                    'Inflacao_decimal': [params['Inflacao']/100], 
                    't': [t_futuro]
                })

                ln_pred = modelo_final.predict(X_futuro)[0]
                pred_consumo = np.exp(ln_pred)
                crescimento = ((pred_consumo - df['Consumo'].iloc[-1]) / df['Consumo'].iloc[-1]) * 100

                projecoes.append({
                    'Cenário': nome,
                    'Consumo': pred_consumo,
                    'Crescimento': crescimento
                })

                st.metric("SELIC", f"{params['Juros']}%")
                st.metric("IPCA", f"{params['Inflacao']}%")
                st.metric("Consumo 2026", f"R$ {pred_consumo:.0f}M", 
                         f"{crescimento:+.1f}%")

                erro_std = np.sqrt(np.mean(modelo_final.resid**2))
                intervalo_inf = np.exp(ln_pred - 1.96*erro_std)
                intervalo_sup = np.exp(ln_pred + 1.96*erro_std)
                st.caption(f"IC 95%: R$ {intervalo_inf:.0f}M - R$ {intervalo_sup:.0f}M")

        # ============================================================================
        # PASSO 4: DOWNLOAD
        # ============================================================================
        st.header("📥 **Passo 4: Baixe seu Relatório**")

        relatorio = f"""
RELATÓRIO ECONOMÉTRICO AUTOMÁTICO
Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}
Por: EconoFácil - Descomplicando a Grana

DADOS ANALISADOS:
• Período: {df['Ano'].min()} - {df['Ano'].max()}
• Observações: {len(df)}

RESULTADOS:
• R² Ajustado: {modelo_final.rsquared_adj:.4f} ({modelo_final.rsquared_adj*100:.1f}%)
• Durbin-Watson: {dw_final:.3f}

PROJEÇÕES 2026:
"""
        for proj in projecoes:
            relatorio += f"• {proj['Cenário']}: R$ {proj['Consumo']:.0f}M ({proj['Crescimento']:+.1f}%)\n"

        relatorio += """
METODOLOGIA:
• Mínimos Quadrados Generalizados (GLS)
• Correção de autocorrelação AR(1)
• Transformação logarítmica

Desenvolvido por Cristiane Graziela
Ciências Econômicas - Anhembi Morumbi
contato@econofacil.com.br
"""

        st.download_button(
            label="📄 **Baixar Relatório (FREE)**",
            data=relatorio,
            file_name=f"relatorio_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

        st.markdown("---")
        st.info("""
        🚀 **Plano Pro (R$29/mês):**
        • Relatórios PDF profissionais
        • 10+ cenários econômicos
        • Suporte prioritário
        • Integração BACEN/IBGE
        """)

        if st.button("💎 **ASSINAR PLANO PRO**"):
            st.balloons()
            st.success("Em breve: contato@econofacil.com.br")

    except Exception as e:
        st.error(f"❌ Erro: {str(e)}")

else:
    st.info("""
    👋 **Bem-vindo ao EconoFácil!**

    Carregue seus dados ou use o exemplo para começar.

    **Você vai receber:**
    • Análise econométrica profissional
    • Projeções para 2026
    • Relatório técnico completo
    • Gráficos interativos
    """)

# ============================================================================
# FORMULÁRIO DE INTERESSE
# ============================================================================
st.markdown("---")
st.header("🚀 **Interessado no Plano Pro?**")

nome = st.text_input("Seu nome:")
email = st.text_input("Seu email:")

if st.button("Quero ser notificado do lançamento Pro!"):
    if nome and email:
        st.balloons()
        st.success(f"Obrigada, {nome}! Você receberá acesso exclusivo ao Plano Pro por R$19 no 1º mês!")
    else:
        st.warning("Por favor, preencha nome e email.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>EconoFácil</strong> - Descomplicando a Grana | 
    Cristiane Graziela | Anhembi Morumbi</p>
    <p>📧 descomplicandoconsutoria@gmail.com | 📱 (11) 96727-3149</p>
</div>
""", unsafe_allow_html=True)




