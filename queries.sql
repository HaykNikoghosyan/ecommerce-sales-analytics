-- Example queries for output/sales.db
-- Run with:  python -m sales_analytics --sql "<query>"
-- or open the file in DB Browser for SQLite / DBeaver.

-- 1. Revenue by category (view)
SELECT * FROM v_category_performance;

-- 2. Monthly revenue with month-over-month growth (window function)
SELECT month,
       ROUND(revenue, 0) AS revenue,
       ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
             / LAG(revenue) OVER (ORDER BY month), 1) AS mom_growth_pct
FROM monthly_summary
ORDER BY month;

-- 3. Top 10 customers and their segment
SELECT customer_id, region, segment, cluster, orders, revenue
FROM v_top_customers
LIMIT 10;

-- 4. Best category in every region (ranking)
SELECT region, category, revenue
FROM (
    SELECT region, category, revenue,
           RANK() OVER (PARTITION BY region ORDER BY revenue DESC) AS rnk
    FROM v_region_category
)
WHERE rnk = 1
ORDER BY revenue DESC;

-- 5. Return rate by channel and price level
SELECT channel,
       CASE WHEN unit_price < 50  THEN '< 50'
            WHEN unit_price < 200 THEN '50-200'
            ELSE '200+' END              AS price_level,
       COUNT(*)                          AS orders,
       ROUND(100.0 * AVG(returned), 2)   AS return_rate_pct
FROM orders
GROUP BY channel, price_level
ORDER BY channel, price_level;

-- 6. At-risk customers worth a win-back campaign
SELECT customer_id, region, recency_days, orders, ROUND(revenue, 2) AS revenue
FROM customers
WHERE segment = 'At Risk'
ORDER BY revenue DESC
LIMIT 20;

-- 7. Month-1 and month-3 retention per cohort
SELECT cohort,
       MAX(CASE WHEN period = 1 THEN ROUND(retention_pct, 1) END) AS m1,
       MAX(CASE WHEN period = 3 THEN ROUND(retention_pct, 1) END) AS m3
FROM cohort_retention
GROUP BY cohort
ORDER BY cohort;

-- 8. Actuals followed by forecast in one timeline
SELECT month, revenue AS value, 'actual' AS kind FROM monthly_summary
UNION ALL
SELECT month, forecast, 'forecast' FROM forecast
ORDER BY month;

-- 9. Busiest weekday/hour combinations
SELECT weekday, hour, COUNT(*) AS orders
FROM orders
GROUP BY weekday, hour
ORDER BY orders DESC
LIMIT 10;

-- 10. Headline KPIs and run information
SELECT * FROM kpis;
SELECT * FROM run_metadata;
