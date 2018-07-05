drop table if exists app.app_ipc_ioa_order_tmp;
create table app.app_ipc_ioa_order_tmp as
select
	dc_id,sku_id,parent_sale_ord_id
FROM app.app_ipc_ioa_whitelist_order WHERE dt BETWEEN '"""+dt60+"""' AND '"""+dt+"""'  ;


--按照85%的订单满足率确定白名单大小及白名单中的SKU（订单量排序算法）
DROP TABLE IF EXISTS app.app_ipc_ioa_whitelist_count_parent_order_rank;
set hive.exec.parallel=true;
set hive.exec.parallel.thread.number=8;
CREATE TABLE app.app_ipc_ioa_whitelist_count_parent_order_rank AS
SELECT
	dc_id
	,parent_sale_ord_id
	,min_sku_parent_ord_cnt
	,row_number() over (partition by dc_id ORDER BY min_sku_parent_ord_cnt DESC) AS parent_order_rank
FROM
(
	-- 订单、订单量
	SELECT
		a.dc_id
		,a.parent_sale_ord_id
		,min(b.sku_parent_ord_cnt) AS min_sku_parent_ord_cnt
	from
			(
				SELECT * FROM app.app_ipc_ioa_order_tmp
		 )a
	JOIN
		(
			-- fdc sku 订单量
			SELECT
				dc_id
				,sku_id
				,COUNT(DISTINCT parent_sale_ord_id) AS sku_parent_ord_cnt
			FROM app.app_ipc_ioa_order_tmp
			GROUP BY
				dc_id,sku_id
		) b
	ON
		a.dc_id = b.dc_id
		AND a.sku_id = b.sku_id
	join
			(--全部为可调拨SKU的单子
			select
  					parent_sale_ord_id
			from
				(select parent_sale_ord_id,count(distinct t1.sku_id) ordersku_cnt,count(distinct t2.sku_id) nullsku_cnt
						 from
						 (SELECT * FROM app.app_ipc_ioa_order_tmp)t1
						 	left join
						 (SELECT * from app.app_ipc_ioa_alloc_total_skulist where dt='"""+dt+"""') t2
							on t1.dc_id = t2.dc_id and t1.sku_id = t2.sku_id
						 group by parent_sale_ord_id
						 	)s where ordersku_cnt=nullsku_cnt) d
			on a.parent_sale_ord_id = d.parent_sale_ord_id
	GROUP BY
		a.dc_id
		,a.parent_sale_ord_id
) c;

insert overwrite TABLE app.app_ipc_ioa_order_rank_whitelist partition(dt='"""+dt+"""')
-- 订单、排序
SELECT DISTINCT
	d.dc_id
	,f.sku_id
	,'0' dc_type
FROM
	(-- fdc、订单、rank（订单量最大-1）
    select t1.* from app.app_ipc_ioa_whitelist_count_parent_order_rank t1
    join
    (select	dc_id from (select s1.dc_id,s2.dc_id as label1,s3.dc_id label2
        from
                (select dc_id from dim.dim_dc_info where dc_type=1) s1
        left join
                (select dc_id from dim.dim_dc_info where dc_type=12) s2
            on s1.dc_id=s2.dc_id
        left join
                (select dc_id from dim.dim_dc_info where dc_type=11) s3
            on s1.dc_id=s3.dc_id
        ) s where label1 is null and  label2 is null )t2
    on t1.dc_id=t2.dc_id
    )d
LEFT JOIN
    (
        -- 计算每个fdc的订单量、需要被满足的订单量
        SELECT
            dc_id
            ,COUNT(DISTINCT parent_sale_ord_id) AS parent_ord_cnt
            ,0.83 * COUNT(DISTINCT parent_sale_ord_id) AS satisfy_parent_ord_cnt
        FROM
            app.app_ipc_ioa_order_tmp
        GROUP BY
            dc_id
    ) e
ON
	d.dc_id = e.dc_id
LEFT JOIN
	app.app_ipc_ioa_order_tmp f
ON
	d.dc_id = f.dc_id
    AND d.parent_sale_ord_id = f.parent_sale_ord_id
WHERE
	d.parent_order_rank <= e.satisfy_parent_ord_cnt

union
	SELECT DISTINCT
	d.dc_id
	,f.sku_id
	,'1' dc_type
FROM
	(-- tdc、订单、rank（订单量最大-1）
    select t1.* from app.app_ipc_ioa_whitelist_count_parent_order_rank t1
    join
    (select dc_id,'1' dc_type from dim.dim_dc_info where dc_type=12 )t2
    on t1.dc_id=t2.dc_id
    )d
LEFT JOIN
    (
        -- 计算每个fdc的订单量、需要被满足的订单量
        SELECT
            dc_id
            ,COUNT(DISTINCT parent_sale_ord_id) AS parent_ord_cnt
            ,0.68 * COUNT(DISTINCT parent_sale_ord_id) AS satisfy_parent_ord_cnt--拍一个数
        FROM
            app.app_ipc_ioa_order_tmp
        GROUP BY
            dc_id
    ) e
ON
	d.dc_id = e.dc_id
LEFT JOIN
	app.app_ipc_ioa_order_tmp f
ON
	d.dc_id = f.dc_id
    AND d.parent_sale_ord_id = f.parent_sale_ord_id
WHERE
	d.parent_order_rank <= e.satisfy_parent_ord_cnt