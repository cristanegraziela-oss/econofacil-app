import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.ar_model import AutoReg
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="EconoFácil - Econometria Simples",
    page_icon="📊",
    layout="wide"
)

# CSS personalizado para interface profissional
st.markdown("""
<style>
    .main .stApp {
        background: linear-gradient(135deg, #f8f9ff 0%, #e8f4fd 100%);
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 15px;
        font-weight: bold;
        font-size: 16px;
        padding: 10px 20px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    .stDownloadButton > button {
        background-color: #2196F3;
        color: white;
        border-radius: 15px;
        font-weight: bold;
        font-size: 16px;
        padding: 12px 24px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMetric > label {
        color: #2E7D32 !important;
        font-weight: bold;
        font-size: 14px;
    }
    .stMetric > div > div {
        color: #1B5E20 !important;
        font-size: 24px;
        font-weight: bold;
    }
    .step-header {
        background: linear-gradient(90deg, #4CAF50, #81C784);
        color: white;
        padding: 15px;
        border-radius: 15px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header principal
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h1 style="color: #2E7D32; font-size: 3.5rem; margin-bottom: 0.5rem;">
            📊 EconoFácil
        </h1>
        <p style="color: #666; font-size: 1.5rem; font-style: italic; margin-bottom: 1rem;">
            Econometria Profissional em 3 Cliques
        </p>
        <p style="color: #2E7D32; font-weight: bold; font-size: 1.2rem;">
            Desenvolvido por Cristiane Graziela - Ciências Econômicas
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Sidebar simplificada
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #f0f8ff; border-radius: 10px; margin-bottom: 2rem;">
        <h3 style="color: #2E7D32;">🚀 Como Funciona</h3>
        <ol style="font-size: 14px; color: #666;">
            <li><strong>1.</strong> Carregue seus dados</li>
            <li><strong>2.</strong> Análise automática</li>
            <li><strong>3.</strong> Veja projeções 2026</li>
            <li><strong>4.</strong> Baixe relatório</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #e8f5e8; border-radius: 10px; margin-bottom: 2rem;">
        <h3 style="color: #2E7D32;">💰 Planos</h3>
        <p><strong>Free:</strong> 1 análise/mês</p>
        <p><strong>Pro:</strong> R$29/mês - Ilimitado</p>
        <p><strong>Business:</strong> R$299/mês - Consultoria</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("📱 WhatsApp", key="whatsapp_sidebar"):
        st.markdown("[Fale comigo!](https://wa.me/5511967273149?text=Olá!%20Testei%20o%20EconoFácil%20e%20gostei%20muito!)", unsafe_allow_html=True)

# ============================================================================
# PASSO 1: UPLOAD SIMPLIFICADO
# ============================================================================
st.markdown('<div class="step-header">📁 Passo 1: Carregue seus Dados</div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader(
        "📁 Escolha Excel ou CSV",
        type=['xlsx', 'csv'],
        help="Precisa ter colunas: Ano, Consumo, Juros, Inflação"
    )

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: #fff3e0; border-radius: 10px; border: 2px dashed #ff9800;">
        <p style="color: #e65100; font-weight: bold; margin: 0;">💡 Dica Rápida</p>
        <p style="color: #666; font-size: 12px; margin: 0.5rem 0 0 0;">Use dados de exemplo para testar!</p>
    </div>
    """, unsafe_allow_html=True)

# Botão grande para dados de exemplo
if st.button("🚀 **TESTAR COM DADOS DE EXEMPLO**", use_container_width=True):
    with st.spinner("🔄 Carregando análise de exemplo..."):
        df_exemplo = pd.DataFrame({
            'Ano': [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
            't': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            'Consumo': [390, 375, 368, 355, 342, 342, 338, 342, 348, 352, 362, 383, 399, 402, 405, 410, 415],
            'Juros': [10, , 12, 13, 14, 14, 14, 13, 14, 14, 14, 14, 13, 15, 13.75, 10.50, 9.00],
            'Inflacao': [87, 86, 85, 82, 79, 78, 78, 75, 75, 75, 76, 72, 76, 79, 4.62, 3.80, 3.50]
        })
        st.session_state.df = df_exemplo
        st.session_state.analise_concluida = True
        st.rerun()

# Carregar dados do usuário
if uploaded_file is not None:
    try:
        with st.spinner("📊 Processando seus dados..."):
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)

            # Detecção automática de colunas
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
                st.warning("⚠️ Selecione as colunas manualmente:")
                col1, col2 = st.columns(2)
                with col1:
                    ano_col = st.selectbox("📅 Ano:", df.columns)
                    cons_col = st.selectbox("💰 Consumo:", df.columns)
                with col2:
                    juros_col = st.selectbox("📈 Juros:", df.columns)
                    infl_col = st.selectbox("📉 Inflação:", df.columns)

                df_padronizado = pd.DataFrame({
                    'Ano': df[ano_col],
                    't': range(1, len(df) + 1),
                    'Consumo': df[cons_col],
                    'Juros': df[juros_col],
                    'Inflacao': df[infl_col]
                })
            else:
                df_padronizado = df.rename(columns=mapeamento)
                df_padronizado['t'] = range(1, len(df_padron) + 1)

            st.session_state.df = df_padronizado
            st.session_state.analise_concluida = True
            st.success(f"✅ Dados carregados: {len(df_padronizado)} observações")
            st.dataframe(df_padronizado.head(5), use_container_width=True)
            st.rerun()

    except Exception as e:
        st.error(f"❌ Erro ao carregar: {str(e)}")

# ============================================================================
# PASSO 2: ANÁLISE (SÓ APARECE SE TIVER DADOS)
# ============================================================================
if 'df' in st.session_state and st.session_state.get('analise_concluida', False):
    st.markdown('<div class="step-header">🔬 Passo 2: Sua Análise Econométrica</div>', unsafe_allow_html=True)

    df = st.session_state.df.copy()

    # Preparar dados
    df['Juros_decimal'] = df['Juros'] / 100
    df['Inflacao_decimal'] = df['Inflacao'] / 100
    df['ln_Consumo'] = np.log(df['Consumo'])

    # Modelo econométrico
    X = df[['Juros_decimal', 'Inflacao_decimal', 't']].copy()
    X = sm.add_constant(X)
    y = df['ln_Consumo'].copy()

    with st.spinner("Executando modelo GLS com correção de autocorrelação..."):
        try:
            # Modelo OLS tradicional
            modelo_trad = sm.OLS(y, X).fit()
            dw_trad = durbin_watson(modelo_trad.resid)

            # Correção AR(1)
            residuos = modelo_trad.resid.values
            ar1_model = AutoReg(residuos, lags=1, old_names=False).fit()
            rho = ar1_model.params[1] if len(ar1_model.params) > 1 else 0.5

            # GLS corrigido
            y_gls = y - rho * y.shift(1)
            X_gls = X - rho * X.shift(1)
            y_gls = y_gls.iloc[1:]
            X_gls = X_gls.iloc[1:]

            from statsmodels.regression.linear_model import GLS
            modelo_final = GLS(y_gls, X_gls).fit()
            dw_final = durbin_watson(modelo_final.resid)

            # Resultados principais
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 R² Ajustado", f"{modelo_final.rsquared_adj:.3}", 
                         f"{modelo_final.rsquared_adj*100:.1f}%")
            with col2:
                st.metric("🔍 Durbin-Watson", f"{dw_final:.3f}", "Melhorado!")
            with col3:
                st.metric("📈 Observações", len(df), "Análise robusta")

            st.success("✅ **Análise concluída com sucesso!** Seu modelo está robusto.")

            # Equação em destaque
            st.markdown("### 📐 **Equação do Seu Modelo**")
            st.latex(rf"\ln(Consumo) = {modelo_final.params[0]:.3f} {modelo_final.params[1]:+.3f} \cdot Juros + {modelo_final.params[2]:+.3f} \cdot Inflação + {modelo_final.params[3]:+.3f} \cdot Tempo")

            st.markdown(f"""
            **💡 Interpretação Rápida:**
            - **Juros**: {modelo_final.params[1]*100:+.2f}% de impacto no consumo            - **Inflação**: {modelo_final.params[2]*100:+.2f}% de impacto no consumo  
            - **Tendência**: +{modelo_final.params[3]*100:.2f}% crescimento anual
            """)

            # Gráficos lado a lado
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📈 **Real vs. Previsto**")
                df['Pred_ln'] = modelo_trad.predict(X)
                df['Pred_Consumo'] = np.exp(df['Pred_ln'])

                fig1 = px.line(df, x='Ano', y=['Consumo', 'Pred_Consumo'], 
                              title="Ajuste do Modelo",
                              labels={'value': 'Consumo (R$ milhões)', 'Ano': 'Ano'},
                              color_discrete_sequence=['#4CAF50', '#2196F3'])
                fig1.update_layout(showlegend=True, font_size=12)
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                st.subheader("📊 **Resíduos do Modelo**")
                fig2 = px.scatter(df, x='Ano', y=modelo_trad.resid, 
                                 title=f"Diagnóstico (DW = {dw_final:.3f})",
                                 labels={'value': 'Resíduos', 'Ano': 'Ano'})
                fig2.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Linha Zero")
                fig2.update_traces(marker=dict(color="#FF5722", size=8))
                st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Erro na análise: {str(e)}")

# ============================================================================
# PASSO 3: PROJEÇÕES (SÓ APARECE SE TIVER ANÁLISE)
# ============================================================================
if 'df' in st.session_state and 'modelo_final' in locals():
    st.markdown('<div class="step-header">🔮 Passo 3: Projeções para 2026</div>', unsafe_allow_html=True)

    # Cenários econômicos
    cenarios = {
        'Base': {'Juros': 8.5, 'Inflacao': 3.25, 'cor': '#4CAF50'},
        'Otimista': {'Juros': 7.5, 'Inflacao': 2.5, 'cor': '#81C784'},
        'Pessimista': {'Juros': 10.5, 'Inflacao': 4.5, 'cor': '#F44336'}
    }

    t_futuro = len(df) + 1
    projecoes = []

    col1, col2, col3 = st.columns(3)

    for i, (nome, params) in enumerate(cenarios.items()):
        with [col1, col2, col3][i]:
           .markdown(f"""
            <div style="background: linear-gradient(135deg, {params['cor']}, #e8f5e8); 
                        padding: 1.5rem; border-radius: 15px; text-align: center; 
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1); height: 100%;">
                <h3 style="color: white; margin: 0 0 1rem 0; font-size: 1.3rem;">
                    {nome.upper()}
                </h3>
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

            st.markdown(f"""
                <div style="background: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                    <p style="margin: 0; font-size: 14px; color: #666;">SELIC</p>
                    <p style="margin: 0; font-size: 18px; font-weight: bold; color: #2E7D32;">
                        {params['Juros']}%
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
                <div style="background: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                    <p style="margin: 0; font-size: 14px; color: #666;">IPCA</p>
                    <p style="margin: 0; font-size: 18px; font-weight: bold; color: #2E7D32;">
                        {params['Inflacao']}%
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Métrica principal
            st.markdown(f"""
                <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; 
                           box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <p style="margin: 0 0 0.5rem 0; font-size: 14px; color: #666;">Consumo 2026</p>
                    <p style="margin: 0; font-size: 28px; font-weight: bold; color: #2E7D32;">
                        R$ {pred_consumo:.0f}M
                    </p>
                    <p style="margin: 0.5rem 0 0 0; font-size: 16px; font-weight: bold; 
                             color: {'green' if crescimento > 0 else 'red'};">
                        {crescimento:+.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Intervalo de confiança
            erro_std = np.sqrt(np.mean(modelo_final.resid**2))
            intervalo_inf = np.exp(ln_pred - 1.96*erro_std)
            intervalo_sup = np.exp(ln_pred + 1.96*erro_std            
            st.markdown(f"""
                <div style="background: #f5f5f5; padding: 0.8rem; border-radius: 8px; 
                           font-size: 12px; color: #666; text-align: center;">
                    📏 IC 95%: R$ {intervalo_inf:.0f}M - R$ {intervalo_sup:.0f}M
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# PASSO 4: DOWNLOAD E PLANO PRO
# ============================================================================
if 'df' in st.session_state and 'modelo_final' in locals():
    st.markdown('<div class="step-header">📥 Passo 4: Seu Relatório Profissional</div>', unsafe_allow_html=True)

    # Gerar relatório
    relatorio = f"""
🚀 RELATÓRIO ECONOMÉTRICO - ECONOFÁCIL
Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}
Por: Cristiane Graziela - Ciências Econômicas

📊 DADOS ANALISADOS:
• Período: {df['Ano'].min()} - {df['Ano'].max()}
• Observações: {len(df)}
• Variáveis: Consumo, Juros (SELIC), Inflação (IPCA)

🔬 RESULTADOS DO MODELO:
• R² Ajustado: {modelo_final.rsquared_adj:.4f} ({modelo_final.rsquared_adj*100:.1f}%)
• Durbin-Watson: {dw_final:.3f} (autocorrelação corrigida)
• Método: GLS com correção AR(1)

📐 EQUAÇÃO ESTIMADA:
ln(Consumo) = {modelo_final.params[0]:.3f} + {modelo_final.params[1]:+.3f} × Juros + 
              {modelo_final.params[2]:+.3f} × Inflação + {modelo_final.params[3]:+.3f} × Tempo

💡 INTERPRETAÇÃO:
• Impacto dos juros: {modelo_final.params[1]*100:+.2f}% no consumo
• Impacto da inflação: {modelo_final.params[2]*100:+.2f}% no consumo
• Tendência de crescimento: +{modelo_final.params[3]*100:.2f}% ao ano

🔮 PROJEÇÕES PARA 2026:

"""

    for proj in projecoes:
        relatorio += f"• {proj['Cenário']}: R$ {proj['Consumo']:.0f}M ({proj['Crescimento']:+.1f}% vs 2025)\n"

    relatorio += f"""
👩‍💼 ANÁLISE POR CRISTIANE GRAZIELA
Ciências Econômicas - Universidade Anhembi Morumbi
contato@econofacil.com.br | (11) 96727-3149

---
EconoFácil - Descomplicando a Grana
www.econofacil.com.br
    """

    # Botão de download grande
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: #e3f2fd; 
                    border-radius: 15px; border: 3px solid #2196F3;">
            <h3 style="color: #1976D2; margin: 0 0 1rem 0;">📄 FREE</h3>
            <p style="color: #666; font-size: 14px; margin: 0;">Relatório básico</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.download_button(
            label="📥 **BAIXAR MEU RELATÓRIO**",
            data=relatorio,
            file_name=f"Relatorio_Econofacil_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    st.markdown("---")

    # Call-to-action Plano Pro
    st.markdown("""
    <div style="background: linear-gradient(135deg, #4CAF50, #81C784); 
                padding: 2rem; border-radius: 15px; text-align: center; 
                color: white; margin: 2rem 0;">
        <h2 style="margin: 0 0 1rem 0; font-size: 2rem;">💎 Quer Mais?</h2>
        <p style="margin: 0 0 1.5rem 0; font-size: 1.2rem;">Plano Pro: Relatórios PDF + 10 cenários + Suporte</p>
        <h3 style="margin: 0 0 1rem 0; font-size: 2.5rem;">Apenas R$29/mês</h3>
        <p style="margin: 0 0 2rem 0; font-size: 1.1rem; opacity 0.9;">
            (R$19 no 1º mês - Lançamento Especial)
        </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("🚀 **ASSINAR PRO AGORA**", use_container_width=True):
            st.balloons()
            st.success("🎉 Em breve! Entre em contato pelo WhatsApp para acesso exclusivo!")

    with col2:
        st.markdown("""
            <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.2); 
                        border-radius: 10px; margin-top: 1rem;">
                <p style="margin: 0; font-size: 1.1rem;">📱 Fale comigo!</p>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.3rem; font-weight: bold;">
                    (11) 96727-3149
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Tela inicial (sem dados)
else:
    st.markdown('<div class="step-header">👋 Bem-vindo ao EconoFácil!</div>', unsafe_allow_html=True)

    col1, col2, col3 = st(3)
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 2rem; background: white; 
                    border-radius: 20px; box-shadow: 0 8px 16px rgba(0,0,0,0.1);">
            <h2 style="color: #2E7D32; margin: 0 0 1rem 0;">📊 O que você vai receber:</h2>
            <ul style="text-align: left; color: #666; font-size: 16px; line-height: 1.6;">
                <li>✅ Análise GLS profissional</li>
                <li>✅ Projeções 2026 (3 cenários)</li>
                <li>✅ Gráficos interativos</li>
                <li>✅ Relatório técnico completo</li>
                <li>✅ Download automático</li>
            </ul>
            <p style="margin: 2rem 0 0 0; color: #2E7D32; font-weight: bold; font-size: 18px;">
                Tudo em menos de 2 minutos! ⏱️
            </p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: #f5f5f5; border-radius: 15px; margin-top: 3rem;">
    <h3 style="color: #2E7D32; margin: 0 0 0.5rem 0;">EconoFácil - Descomplicando a Grana</h3>
    <p style="color: #666; margin: 0 0 1rem 0; font-size: 16px;">
        Desenvolvido por <strong>Cristiane Graziela</strong> | Ciências Econômicas - Anhembi Morumbi
    </p>
    <p style="color: #666; margin: 0; font-size: 14px;">
        📧 contato@econofacil.com.br | 📱 (11) 96727-3149
    </p>
</div>
""", unsafe_allow_html=True)





