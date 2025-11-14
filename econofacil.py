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

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="EconoFácil - Econometria Profissional",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS PERSONALIZADO
# ============================================================================
st.markdown("""
<style>
    /* Fundo gradiente */
    .main .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Cards brancos */
    .stApp > div {
        background-color: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }

    /* Métricas */
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .stMetric > label {
        color: #667eea !important;
        font-weight: bold;
        font-size: 16px;
    }

    /* Botões */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        font-weight: bold;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 16px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border-radius: 10px;
        font-weight: bold;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 16px;
    }

    /* Headers */
    h1, h2, h3 {
        color: #667eea;
    }

    /* Info boxes */
    .stInfo {
        background-color: #e7f3ff;
        border-left: 4px solid #2196F3;
    }

    /* Success boxes */
    .stSuccess {
        background-color: #e8f5e9;
        border-left: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER PROFISSIONAL
# ============================================================================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem; color: #667eea;">
            📊 EconoFácil
        </h1>
        <p style="font-size: 1.4rem; color: #666; font-style: italic; margin-bottom: 0.5rem;">
            Econometria Profissional em 3 Cliques
        </p>
        <p style="font-size: 1.1rem; color: #667eea; font-weight: bold;">
            Desenvolvido por Cristiane Graziela - Ciências Econômicas
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### 🎯 **Como Funciona**")
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <p style="margin: 0.5rem 0;"><strong>1️⃣ Upload</strong><br>Carregue Excel/CSV com seus dados</p>
        <p style="margin: 0.5rem 0;"><strong>2️⃣ Análise</strong><br>Modelo GLS automático</p>
        <p style="margin: 0.5rem 0;"><strong>3️⃣ Projeções</strong><br>3 cenários para 2026</p>
        <p style="margin: 0.5rem 0;"><strong>4️⃣ Download</strong><br>Relatório completo</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 💰 **Planos**")
    st.markdown("""
    <div style="background-color: #e8f5e9; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <p style="margin: 0.5rem 0;"><strong>🆓 Free</strong><br>Análise básica + 1 relatório/mês</p>
        <p style="margin: 0.5rem 0;"><strong>⭐ Pro - R$29/mês</strong><br>Relatórios ilimitados + suporte</p>
        <p style="margin: 0.5rem 0;"><strong>💎 Business - R$299/mês</strong><br>Consultoria personalizada</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 💬 **Contato**")
    whatsapp_link = "https://wa.me/5511967273149?text=Olá!%20Testei%20o%20EconoFácil%20e%20gostei%20muito!"
    st.markdown(f"[📱 WhatsApp Business]({whatsapp_link})")

    st.markdown("---")
    st.caption("© 2024 EconoFácil | Anhembi Morumbi")

# ============================================================================
# PASSO 1: UPLOAD DE DADOS
# ============================================================================
st.markdown("## 📁 **Passo 1: Carregue seus Dados**")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
        <p style="margin-bottom: 0.5rem;"><strong>📋 Formato esperado:</strong></p>
        <ul style="margin-top: 0;">
            <li><strong>Ano</strong>: 2009, 2010, 2011...</li>
            <li><strong>Consumo</strong>: Valores em R$ milhões</li>
            <li><strong>Juros</strong>: Taxa Selic em %</li>
            <li><strong>Inflacao</strong>: IPCA em %</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Arraste seu arquivo aqui ou clique para selecionar",
        type=['xlsx', 'csv'],
        help="Aceita Excel (.xlsx) ou CSV (.csv)"
    )

with col2:
    st.markdown("""
    <div style="background-color: #e7f3ff; padding: 1.5rem; border-radius: 10px; text-align: center;">
        <p style="font-size: 3rem; margin: 0;">💡</p>
        <p style="margin: 0.5rem 0;"><strong>Não tem dados?</strong></p>
        <p style="margin: 0; color: #666;">Use nosso exemplo para testar!</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 **Usar Dados de Exemplo**", use_container_width=True):
        df_exemplo = pd.DataFrame({
            'Ano': [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
            't': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            'Consumo': [390, 375, 368, 355, 342, 342, 338, 342, 348, 352, 362, 383, 399, 402, 405, 410, 415],
            'Juros': [10, 11, 12, 13, 14, 14, 14, 13, 14, 14, 14, 14, 13, 15, 13.75, 10.50, 9.00],
            'Inflacao': [87, 86, 85, 82, 79, 78, 78, 75, 75, 75, 76, 72, 76, 79, 4.62, 3.80, 3.50]
        })
        st.session_state.df = df_exemplo
        st.success("✅ Dados de exemplo carregados com sucesso!")
        st.balloons()

# Processar upload
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        st.success(f"✅ Arquivo carregado: {len(df)} observações")

        # Detectar colunas
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

        if len(mapeamento) >= 4:
            df_padronizado = df.rename(columns=mapeamento)
            df_padronizado['t'] = range(1, len(df_padronizado) + 1)
            st.session_state.df = df_padronizado
            st.dataframe(df_padronizado.head(), use_container_width=True)
        else:
            st.warning("⚠️ Não consegui detectar as colunas automaticamente. Selecione manualmente:")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                ano_col = st.selectbox("Coluna do Ano:", df.columns)
            with col2:
                cons_col = st.selectbox("Coluna do Consumo:", df.columns)
            with col3:
                juros_col = st.selectbox("Coluna dos Juros:", df.columns)
            with col4:
                infl_col = st.selectbox("Coluna da Inflação:", df.columns)

            if st.button("✅ Confirmar Seleção"):
                df_padronizado = pd.DataFrame({
                    'Ano': df[ano_col],
                    't': range(1, len(df) + 1),
                    'Consumo': df[cons_col],
                    'Juros': df[juros_col],
                    'Inflacao': df[infl_col]
                })
                st.session_state.df = df_padronizado
                st.success("✅ Dados configurados!")
                st.rerun()

    except Exception as e:
        st.error(f"❌ Erro ao carregar arquivo: {str(e)}")

# ============================================================================
# PASSO 2: ANÁLISE
# ============================================================================
if 'df' in st.session_state:
    st.markdown("---")
    st.markdown("## 🔬 **Passo 2: Análise Econométrica**")

    df = st.session_state.df.copy()

    # Preparar dados
    df['Juros_decimal'] = df['Juros'] / 100
    df['Inflacao_decimal'] = df['Inflacao'] / 100
    df['ln_Consumo'] = np.log(df['Consumo'])

    X = df[['Juros_decimal', 'Inflacao_decimal', 't']].copy()
    X = sm.add_constant(X)
    y = df['ln_Consumo'].copy()

    with st.spinner('🔄 Processando modelo econométrico...'):
        try:
            # Modelo OLS
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

            # Resultados
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "📊 R² Ajustado",
                    f"{modelo_final.rsquared_adj:.1%}",
                    f"{modelo_final.rsquared_adj*100:.1f}% de explicação"
                )

            with col2:
                st.metric(
                    "🎯 Durbin-Watson",
                    f"{dw_final:.3f}",
                    "Autocorrelação corrigida" if 1.5 < dw_final < 2.5 else "Atenção"
                )

            with col3:
                st.metric(
                    "📈 Observações",
                    f"{len(df)}",
                    f"{df['Ano'].min()} - {df['Ano'].max()}"
                )

            # Equação
            st.markdown("### 📐 **Equação Estimada**")
            st.latex(f"\\ln(Consumo) = {modelo_final.params[0]:.3f} {modelo_final.params[1]:+.3f} \\cdot Juros {modelo_final.params[2]:+.3f} \\cdot Inflação {modelo_final.params[3]:+.3f} \\cdot Tempo")

            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin-top: 1rem;">
                <p style="margin: 0.5rem 0;"><strong>📊 Interpretação Econômica:</strong></p>
                <ul style="margin-top: 0.5rem;">
                    <li>Cada <strong>1 p.p. ↑ nos juros</strong> → Consumo <strong>{modelo_final.params[1]*100:+.2f}%</strong></li>
                    <li>Cada <strong>1 p.p. ↑ na inflação</strong> → Consumo <strong>{modelo_final.params[2]*100:+.2f}%</strong></li>
                    <li><strong>Tendência anual</strong>: <strong>{modelo_final.params[3]*100:+.2f}%</strong> no consumo</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            # Gráficos
            st.markdown("### 📈 **Visualizações**")

            col1, col2 = st.columns(2)

            with col1:
                df['Pred_ln'] = modelo_trad.predict(X)
                df['Pred_Consumo'] = np.exp(df['Pred_ln'])

                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=df['Ano'], y=df['Consumo'],
                    mode='lines+markers',
                    name='Real',
                    line=dict(color='#667eea', width=3)
                ))
                fig1.add_trace(go.Scatter(
                    x=df['Ano'], y=df['Pred_Consumo'],
                    mode='lines',
                    name='Previsto',
                    line=dict(color='#764ba2', width=2, dash='dash')
                ))
                fig1.update_layout(
                    title="Real vs Previsto",
                    xaxis_title="Ano",
                    yaxis_title="Consumo (R$ milhões)",
                    hovermode='x unified'
                )
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=df['Ano'], y=modelo_trad.resid,
                    mode='markers',
                    marker=dict(size=10, color='#667eea'),
                    name='Resíduos'
                ))
                fig2.add_hline(y=0, line_dash="dash", line_color="red")
                fig2.update_layout(
                    title=f"Resíduos (DW = {dw_final:.3f})",
                    xaxis_title="Ano",
                    yaxis_title="Resíduos",
                    hovermode='x'
                )
                st.plotly_chart(fig2, use_container_width=True)

            # ============================================================================
            # PASSO 3: PROJEÇÕES
            # ============================================================================
            st.markdown("---")
            st.markdown("## 🔮 **Passo 3: Projeções 2026**")

            cenarios = {
                'Base': {'Juros': 8.5, 'Inflacao': 3.25, 'cor': '#667eea'},
                'Otimista': {'Juros': 7.5, 'Inflacao': 2.5, 'cor': '#11998e'},
                'Pessimista': {'Juros': 10.5, 'Inflacao': 4.5, 'cor': '#eb3b5a'}
            }

            t_futuro = len(df) + 1
            projecoes = []

            col1, col2, col3 = st.columns(3)

            for i, (nome, params) in enumerate(cenarios.items()):
                with [col1, col2, col3][i]:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {params['cor']} 0%, {params['cor']}dd 100%); 
                                padding: 1.5rem; border-radius: 15px; color: white; text-align: center;">
                        <h3 style="color: white; margin-bottom: 1rem;">{nome.upper()}</h3>
                    </div>
                    """, unsafe_allow_html=True)

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

                    st.metric("🎯 SELIC", f"{params['Juros']}%")
                    st.metric("📊 IPCA", f"{params['Inflacao']}%")
                    st.metric("💰 Consumo 2026", f"R$ {pred_consumo:.0f}M", f"{crescimento:+.1f}%")

                    erro_std = np.sqrt(np.mean(modelo_final.resid**2))
                    intervalo_inf = np.exp(ln_pred - 1.96*erro_std)
                    intervalo_sup = np.exp(ln_pred + 1.96*erro_std)
                    st.caption(f"IC 95%: R$ {intervalo_inf:.0f}M - R$ {intervalo_sup:.0f}M")

            # ============================================================================
            # PASSO 4: DOWNLOAD
            # ============================================================================
            st.markdown("---")
            st.markdown("## 📥 **Passo 4: Baixe seu Relatório**")

            relatorio = f"""
╔══════════════════════════════════════════════════════════════╗
║           RELATÓRIO ECONOMÉTRICO AUTOMÁTICO                  ║
║                    EconoFácil                                ║
╚══════════════════════════════════════════════════════════════╝

📅 Gerado em: {datetime.now().strftime('%d/%m/%Y às %H:%M')}
👤 Desenvolvido por: Cristiane Graziela - Ciências Econômicas

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 DADOS ANALISADOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Período: {df['Ano'].min()} - {df['Ano'].max()}
• Observações: {len(df)} anos
• Variáveis: Consumo, Juros (Selic), Inflação (IPCA)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔬 RESULTADOS DO MODELO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• R² Ajustado: {modelo_final.rsquared_adj:.4f} ({modelo_final.rsquared_adj*100:.1f}%)
• Durbin-Watson: {dw_final:.3f}
• Método: Mínimos Quadrados Generalizados (GLS)
• Correção: Autocorrelação AR(1) com ρ = {rho:.3f}

📐 EQUAÇÃO ESTIMADA:
ln(Consumo) = {modelo_final.params[0]:.3f} {modelo_final.params[1]:+.3f}×Juros {modelo_final.params[2]:+.3f}×Inflação {modelo_final.params[3]:+.3f}×Tempo

📊 INTERPRETAÇÃO ECONÔMICA:
• Impacto dos Juros: {modelo_final.params[1]*100:+.2f}% no consumo (cada 1 p.p.)
• Impacto da Inflação: {modelo_final.params[2]*100:+.2f}% no consumo (cada 1 p.p.)
• Tendência Temporal: {modelo_final.params[3]*100:+.2f}% ao ano

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔮 PROJEÇÕES PARA 2026
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            for proj in projecoes:
                relatorio += f"• {proj['Cenário']:12s}: R$ {proj['Consumo']:6.0f}M ({proj['Crescimento']:+.1f}%)\n"

            relatorio += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 METODOLOGIA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Transformação logarítmica da variável dependente
2. Estimação OLS inicial para detectar autocorrelação
3. Correção GLS com estrutura AR(1)
4. Teste Durbin-Watson para validação
5. Projeções com intervalos de confiança de 95%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📧 CONTATO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EconoFácil - Descomplicando a Grana
Cristiane Graziela - Ciências Econômicas
Universidade Anhembi Morumbi

📧 contato@econofacil.com.br
📱 (11) 96727-3149
🌐 https://econofacil-app.streamlit.app

© 2024 EconoFácil - Todos os direitos reservados
"""

            col1, col2 = st.columns([2, 1])

            with col1:
                st.download_button(
                    label="📄 **BAIXAR RELATÓRIO COMPLETO (FREE)**",
                    data=relatorio,
                    file_name=f"EconoFacil_Relatorio_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            with col2:
                if st.button("💎 **ASSINAR PLANO PRO**", use_container_width=True):
                    st.balloons()
                    st.success("🎉 Em breve: Relatórios PDF + 10 cenários!")

            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-top: 2rem;">
                <h3 style="color: white; margin-bottom: 1rem;">🚀 Plano Pro - R$29/mês</h3>
                <p style="margin: 0.5rem 0;">✅ Relatórios PDF profissionais com gráficos</p>
                <p style="margin: 0.5rem 0;">✅ 10+ cenários econômicos personalizados</p>
                <p style="margin: 0.5rem 0;">✅ Suporte prioritário via WhatsApp</p>
                <p style="margin: 0.5rem 0;">✅ Integração automática BACEN/IBGE</p>
                <p style="margin: 0.5rem 0;">✅ Histórico de análises ilimitado</p>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Erro na análise: {str(e)}")
            st.info("💡 Verifique se seus dados estão no formato correto e tente novamente.")

else:
    # Tela inicial
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 3rem; border-radius: 20px; color: white; text-align: center; margin: 2rem 0;">
        <h2 style="color: white; margin-bottom: 1rem;">👋 Bem-vindo ao EconoFácil!</h2>
        <p style="font-size: 1.2rem; margin-bottom: 2rem;">
            Transforme seus dados econômicos em insights profissionais em minutos
        </p>
        <div style="background-color: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 10px; margin-top: 1rem;">
            <p style="margin: 0.5rem 0;"><strong>✨ Você vai receber:</strong></p>
            <p style="margin: 0.5rem 0;">📊 Análise econométrica GLS robusta</p>
            <p style="margin: 0.5rem 0;">🔮 Projeções para 2026 (3 cenários)</p>
            <p style="margin: 0.5rem 0;">📈 Gráficos interativos profissionais</p>
            <p style="margin: 0.5rem 0;">📄 Relatório técnico completo</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FORMULÁRIO DE INTERESSE
# ============================================================================
st.markdown("---")
st.markdown("## 🚀 **Interessado no Plano Pro?**")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    with st.form("formulario_interesse"):
        nome = st.text_input("📝 Seu nome completo:")
        email = st.text_input("📧 Seu melhor email:")
        interesse = st.selectbox(
            "💼 Você é:",
            ["Estudante de Economia", "Professor/Pesquisador", "Consultor", "Empresário/Gestor", "Outro"]
        )

        submitted = st.form_submit_button("✨ **QUERO SER NOTIFICADO!**", use_container_width=True)

        if submitted:
            if nome and email:
                st.balloons()
                st.success(f"""
                🎉 **Obrigada, {nome}!**

                Você receberá:
                • Acesso exclusivo ao Plano Pro por **R$19 no 1º mês** (33% OFF)
                • Convite para o grupo VIP no WhatsApp
                • E-book gratuito "Econometria Aplicada"

                📧 Fique de olho no email: {email}
                """)
            else:
                st.warning("⚠️ Por favor, preencha nome e email.")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
        <strong>📊 EconoFácil</strong> - Descomplicando a Grana
    </p>
    <p style="margin: 0.5rem 0;">
        Desenvolvido por <strong>Cristiane Graziela</strong><br>
        Ciências Econômicas - Universidade Anhembi Morumbi
    </p>
    <p style="margin: 1rem 0;">
        📧 contato@econofacil.com.br | 
        📱 (11) 96727-3149 | 
        🌐 <a href="https://econofacil-app.streamlit.app" style="color: #667eea;">econofacil-app.streamlit.app</a>
    </p>
    <p style="color: #999; font-size: 0.9rem;">
        © 2024 EconoFácil - Todos os direitos reservados
    </p>
</div>
""", unsafe_allow_html=True)

