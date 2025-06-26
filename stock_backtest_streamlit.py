import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import io
import plotly.express as px
import plotly.graph_objects as go
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Stock Backtesting Tool",
    page_icon="📈",
    layout="wide"
)

def backtest_trades(df):
    """
    Backtest stock trades from DataFrame using Yahoo Finance data
    
    Args:
        df (pd.DataFrame): DataFrame containing trade data
    
    Returns:
        pd.DataFrame: DataFrame with backtest results
    """
    
    # Rename columns as specified
    df = df.rename(columns={
        'CMP (₹)': 'Entry_Price',
        'Target 1 (₹)': 'Target', 
        'Stoploss (₹)': 'Stoploss'
    })
    
    # Parse Entry_Date as datetime
    df['Entry_Date'] = pd.to_datetime(df['Entry_Date'], format='%d-%m-%Y')
    
    # Initialize result columns
    df['Success'] = 'None'
    df['Hit_Date'] = None
    df['Exit_Price'] = None
    df['Returns_%'] = None
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process each trade
    for idx, row in df.iterrows():
        symbol = row['Symbol']
        entry_date = row['Entry_Date']
        entry_price = row['Entry_Price']
        target = row['Target']
        stoploss = row['Stoploss']
        
        # Update progress
        progress = (idx + 1) / len(df)
        progress_bar.progress(progress)
        status_text.text(f'Processing {symbol} (Trade {idx+1}/{len(df)})')
        
        try:
            # Create Yahoo Finance symbol (NSE stocks need .NS suffix)
            yf_symbol = f"{symbol}.NS"
            
            # Calculate date range for fetching data
            start_date = entry_date + timedelta(days=1)
            end_date = entry_date + timedelta(days=11)  # +11 to get 10 trading days
            
            # Fetch OHLCV data from Yahoo Finance
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                continue
                
            # Check each candle for target or stoploss hit
            hit_found = False
            
            for date, candle in data.iterrows():
                high = candle['High']
                low = candle['Low']
                
                # Check if target is hit first (High >= Target)
                if high >= target:
                    df.at[idx, 'Success'] = 'Success'
                    df.at[idx, 'Hit_Date'] = date.strftime('%Y-%m-%d')
                    df.at[idx, 'Exit_Price'] = target
                    df.at[idx, 'Returns_%'] = ((target - entry_price) / entry_price) * 100
                    hit_found = True
                    break
                
                # Check if stoploss is hit (Low <= Stoploss)
                elif low <= stoploss:
                    df.at[idx, 'Success'] = 'Failure'
                    df.at[idx, 'Hit_Date'] = date.strftime('%Y-%m-%d')
                    df.at[idx, 'Exit_Price'] = stoploss
                    df.at[idx, 'Returns_%'] = ((stoploss - entry_price) / entry_price) * 100
                    hit_found = True
                    break
                
        except Exception as e:
            continue
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return df

def generate_summary(df):
    """
    Generate summary statistics for backtest results
    
    Args:
        df (pd.DataFrame): DataFrame with backtest results
    
    Returns:
        dict: Summary statistics
    """
    
    # Filter out 'None' results for success rate calculation
    considered_trades = df[df['Success'].isin(['Success', 'Failure'])]
    
    if len(considered_trades) == 0:
        return None
    
    # Calculate overall statistics
    total_trades = len(considered_trades)
    successful_trades = len(considered_trades[considered_trades['Success'] == 'Success'])
    failed_trades = len(considered_trades[considered_trades['Success'] == 'Failure'])
    no_result_trades = len(df[df['Success'] == 'None'])
    
    overall_success_rate = (successful_trades / total_trades) * 100 if total_trades > 0 else 0
    
    # Calculate returns
    avg_return = considered_trades['Returns_%'].mean() if len(considered_trades) > 0 else 0
    total_return = considered_trades['Returns_%'].sum() if len(considered_trades) > 0 else 0
    
    summary = {
        'total_trades': total_trades,
        'successful_trades': successful_trades,
        'failed_trades': failed_trades,
        'no_result_trades': no_result_trades,
        'overall_success_rate': overall_success_rate,
        'avg_return': avg_return,
        'total_return': total_return
    }
    
    # Score-wise success rates
    score_summary = []
    if 'Score' in df.columns:
        for score in sorted(df['Score'].unique()):
            score_trades = considered_trades[considered_trades['Score'] == score]
            if len(score_trades) > 0:
                score_success = len(score_trades[score_trades['Success'] == 'Success'])
                score_success_rate = (score_success / len(score_trades)) * 100
                score_avg_return = score_trades['Returns_%'].mean()
                
                score_summary.append({
                    'Score': score,
                    'Total_Trades': len(score_trades),
                    'Successful_Trades': score_success,
                    'Success_Rate': score_success_rate,
                    'Avg_Return_%': score_avg_return
                })
    
    return summary, pd.DataFrame(score_summary)

def create_visualizations(df, score_summary):
    """Create visualizations for the backtest results"""
    
    # Success Rate by Score
    if not score_summary.empty:
        fig1 = px.bar(
            score_summary, 
            x='Score', 
            y='Success_Rate',
            title='Success Rate by Score',
            labels={'Success_Rate': 'Success Rate (%)', 'Score': 'Score'},
            color='Success_Rate',
            color_continuous_scale='RdYlGn'
        )
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    # Returns Distribution
    considered_trades = df[df['Success'].isin(['Success', 'Failure'])]
    if len(considered_trades) > 0:
        fig2 = px.histogram(
            considered_trades,
            x='Returns_%',
            color='Success',
            title='Returns Distribution',
            labels={'Returns_%': 'Returns (%)', 'count': 'Number of Trades'},
            color_discrete_map={'Success': 'green', 'Failure': 'red'}
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Success/Failure Pie Chart
    success_counts = df['Success'].value_counts()
    if len(success_counts) > 0:
        fig3 = px.pie(
            values=success_counts.values,
            names=success_counts.index,
            title='Trade Outcomes Distribution',
            color_discrete_map={'Success': 'green', 'Failure': 'red', 'None': 'gray'}
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

def main():
    """Main Streamlit app"""
    
    st.title("📈 Stock Backtesting Tool")
    st.markdown("Upload your CSV file containing trade data to backtest your trading strategy")
    
    # Sidebar
    st.sidebar.header("Instructions")
    st.sidebar.markdown("""
    **CSV Format Required:**
    - Symbol: Stock symbol (without .NS)
    - Entry_Date: Date in DD-MM-YYYY format
    - CMP (₹): Entry price
    - Target 1 (₹): Target price
    - Stoploss (₹): Stop loss price
    - Score: (Optional) Trade score
    
    **Example:**
    ```
    Symbol,Entry_Date,CMP (₹),Target 1 (₹),Stoploss (₹),Score
    RELIANCE,15-01-2024,2500,2600,2400,8
    TCS,16-01-2024,3200,3300,3100,9
    ```
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your trade data CSV file"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Display uploaded data
            st.subheader("📋 Uploaded Data Preview")
            st.dataframe(df.head(10))
            
            st.info(f"Total trades in file: {len(df)}")
            
            # Validate required columns
            required_columns = ['Symbol', 'Entry_Date', 'CMP (₹)', 'Target 1 (₹)', 'Stoploss (₹)']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                st.stop()
            
            # Run backtest button
            if st.button("🚀 Run Backtest", type="primary"):
                st.subheader("🔄 Running Backtest...")
                
                # Run backtest
                with st.spinner("Processing trades..."):
                    result_df = backtest_trades(df.copy())
                
                # Generate summary
                summary_data = generate_summary(result_df)
                
                if summary_data is None:
                    st.error("❌ No trades could be evaluated (no target/stoploss hits)")
                    st.stop()
                
                summary, score_summary = summary_data
                
                # Display results
                st.subheader("📊 Backtest Results")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Trades", summary['total_trades'])
                
                with col2:
                    st.metric("Success Rate", f"{summary['overall_success_rate']:.1f}%")
                
                with col3:
                    st.metric("Avg Return", f"{summary['avg_return']:.2f}%")
                
                with col4:
                    st.metric("Total Return", f"{summary['total_return']:.2f}%")
                
                # Additional metrics
                col5, col6, col7 = st.columns(3)
                
                with col5:
                    st.metric("Successful Trades", summary['successful_trades'], 
                             delta=f"{(summary['successful_trades']/summary['total_trades']*100):.1f}%")
                
                with col6:
                    st.metric("Failed Trades", summary['failed_trades'],
                             delta=f"{(summary['failed_trades']/summary['total_trades']*100):.1f}%")
                
                with col7:
                    st.metric("No Result", summary['no_result_trades'])
                
                # Visualizations
                st.subheader("📈 Analysis Charts")
                create_visualizations(result_df, score_summary)
                
                # Score-wise summary table
                if not score_summary.empty:
                    st.subheader("📋 Score-wise Performance")
                    st.dataframe(score_summary, use_container_width=True)
                
                # Detailed results
                st.subheader("📄 Detailed Results")
                
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    filter_success = st.selectbox(
                        "Filter by Result:",
                        ["All", "Success", "Failure", "None"],
                        index=0
                    )
                
                with col2:
                    if 'Score' in result_df.columns:
                        filter_score = st.selectbox(
                            "Filter by Score:",
                            ["All"] + sorted(result_df['Score'].unique().tolist()),
                            index=0
                        )
                    else:
                        filter_score = "All"
                
                # Apply filters
                filtered_df = result_df.copy()
                if filter_success != "All":
                    filtered_df = filtered_df[filtered_df['Success'] == filter_success]
                if filter_score != "All" and 'Score' in result_df.columns:
                    filtered_df = filtered_df[filtered_df['Score'] == filter_score]
                
                st.dataframe(filtered_df, use_container_width=True)
                
                # Download results
                st.subheader("💾 Download Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Convert to CSV
                    csv_buffer = io.StringIO()
                    result_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="📥 Download Detailed Results (CSV)",
                        data=csv_data,
                        file_name=f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    if not score_summary.empty:
                        # Convert score summary to CSV
                        score_csv_buffer = io.StringIO()
                        score_summary.to_csv(score_csv_buffer, index=False)
                        score_csv_data = score_csv_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Download Score Summary (CSV)",
                            data=score_csv_data,
                            file_name=f"score_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                st.success("✅ Backtesting completed successfully!")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    else:
        st.info("👆 Please upload a CSV file to start backtesting")
        
        # Show sample data format
        st.subheader("📋 Sample CSV Format")
        sample_data = {
            'Symbol': ['RELIANCE', 'TCS', 'INFY'],
            'Entry_Date': ['15-01-2024', '16-01-2024', '17-01-2024'],
            'CMP (₹)': [2500, 3200, 1400],
            'Target 1 (₹)': [2600, 3300, 1450],
            'Stoploss (₹)': [2400, 3100, 1350],
            'Score': [8, 9, 7]
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)

if __name__ == "__main__":
    main()