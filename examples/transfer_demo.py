"""Original Recall API demonstration with an explicit offline model fixture."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import SCENARIO
from recall_core import Recall

class OfflineModel:
    calls = 0
    def decide(self, text, features):
        self.calls += 1
        action = 'Roll back service Alpha to the previous version' if self.calls == 1 else 'Inspect the error logs for service Beta'
        return dict(features=features, action=action, reason='Scripted offline fixture; no provider inference.')

model = OfflineModel()
core = Recall(model)
first = core.decide(SCENARIO['situation_a'])
assert first['source'] == 'MODEL'
core.save_outcome(first['id'], True, 0.8, SCENARIO['outcome']['feedback'])
second = core.decide(SCENARIO['situation_b'])
assert second['source'] == 'EXPERIENCE' and model.calls == 1
assert second['action'] == 'Roll back service Beta to the previous version'
blocked = core.decide(SCENARIO['situation_b'].replace('Rollback is available', 'Rollback is unavailable'))
assert blocked['source'] == 'MODEL' and model.calls == 2
assert blocked['evidence']['status'] == 'TRANSFER BLOCKED: critical mismatch'
print('First decision:', first['source'])
print('After reported success:', second['source'], '-', second['action'])
print('Explicit prohibition:', blocked['evidence']['status'], '->', blocked['source'])
print('PASS: original Recall API; scripted model, zero network calls.')
