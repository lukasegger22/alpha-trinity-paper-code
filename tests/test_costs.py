import pandas as pd
from ai_ls_allocation.engine.costs import trade_costs, borrow_costs

def test_trade_costs_and_borrow_behave():
    cfg = dict(commission_bps=0.2, half_spread_bps=1.5, slippage_extra_bps=1.0, short_borrow_annual_bps=150)
    delta = pd.DataFrame([[0.1,-0.1],[0.0,0.2],[-0.05,0.05]], columns=["A","B"])
    tc = trade_costs(delta, cfg)
    assert (tc >= 0).all()
    expected_bps = (0.2+1.5+1.0)/10000.0
    turn = delta.abs().sum(axis=1)
    assert (abs(tc - expected_bps*turn) < 1e-12).all()

    w = pd.DataFrame([[0.2,-0.1],[0.0,-0.3],[-0.05,0.0]], columns=["A","B"])
    bc = borrow_costs(w, cfg)
    rate = 150/10000/252.0
    expected_bc = rate * w.clip(upper=0).abs().sum(axis=1)
    assert (bc >= 0).all()
    assert (abs(bc - expected_bc) < 1e-12).all()
