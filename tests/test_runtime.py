"""Original Recall regression cases; scripted models and mocked transport, no live calls."""
from pathlib import Path
import sys
import json
import os
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import SCENARIO, create_app
from model_client import NebiusClient, ModelError
from recall_core import Recall, critical_mismatch, features_for, similarity

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

class FakeModel:
    ready = True

    def __init__(self):
        self.calls = 0

    def decide(self, text, features):
        self.calls += 1
        return dict(features=features, action='Вернуться к предыдущей стабильной версии сервиса.',
                    reason='Проблема появилась после обновления; проверьте восстановление задержки после отката.')


class TransferObjectRegression(unittest.TestCase):
    def transfer(self, old, new, action):
        original = f'После обновления сервис {old} отвечает медленно, очередь запросов растёт. Доступен откат версии.'
        current = f'После обновления сервис {new} отвечает медленно, очередь запросов растёт. Доступен откат версии.'
        model = FakeModel()
        with patch.object(model, 'decide', return_value=dict(
                features=features_for(original), action=action, reason='Проверенное действие.')) as call:
            core = Recall(model)
            first = core.decide(original)
            experience = core.save_outcome(first['id'], True, .8, 'Демонстрационный исход')
            result = core.decide(current)
            self.assertEqual(call.call_count, 1)
        self.assertEqual(result['source'], 'EXPERIENCE')
        self.assertEqual(result['evidence']['experience'], experience)
        self.assertEqual(result['evidence']['similarity'],
                         similarity(features_for(original), features_for(current)))
        self.assertEqual(result['evidence']['action'], result['action'])
        self.assertEqual(core.snapshot()['experiences'][0], experience)
        self.assertEqual(core.snapshot()['decisions'][0], first)
        return result['action']

    def test_named_service_transfer(self):
        for old, new in [('Альфа', 'Бета'), ('Орион', 'Вега'),
                         ('Payments-v2', 'Billing-v3'), ('«Север API»', '«Юг API»')]:
            with self.subTest(old=old, new=new):
                self.assertEqual(self.transfer(old, new,
                    f'Откатить версию сервиса {old} до предыдущей'),
                    f'Откатить версию сервиса {new} до предыдущей')

    def test_ambiguous_object_keeps_action(self):
        action = 'Откатить версию сервиса Альфа до предыдущей'
        for new in ['Бета и Гамма', 'Бета или сервис Гамма', 'без имени', 'Бета Север']:
            with self.subTest(new=new):
                self.assertEqual(self.transfer('Альфа', new, action), action)
        self.assertEqual(self.transfer('Альфа и Гамма', 'Бета и Гамма', action), action)

    def test_other_action_text_preserved(self):
        action = 'Откатить версию сервиса Альфа до предыдущей; сохранить журнал Альфа.'
        self.assertEqual(self.transfer('Альфа', 'Бета', action),
                         'Откатить версию сервиса Бета до предыдущей; сохранить журнал Альфа.')
        generic = 'Вернуться к предыдущей стабильной версии сервиса.'
        self.assertEqual(self.transfer('Альфа', 'Бета', generic), generic)
        ambiguous = 'Откатить версию сервиса Альфа Север до предыдущей'
        self.assertEqual(self.transfer('Альфа', 'Бета', ambiguous), ambiguous)


class CriticalMismatchRegression(unittest.TestCase):
    original = 'После обновления сервис Орион отвечает медленно, очередь запросов растёт. Доступен откат версии.'
    current = original.replace('Орион', 'Вега').replace('Доступен откат версии', 'Откат версии недоступен')
    action = 'Откатить версию сервиса Орион до предыдущей'
    fallback = 'Проверить журнал ошибок сервиса Вега'
    status = 'TRANSFER BLOCKED: critical mismatch'

    def model(self):
        owner = self
        class ScriptedModel:
            ready = True
            calls = 0
            def decide(self, text, features):
                self.calls += 1
                return dict(features=features, action=owner.action if self.calls == 1 else owner.fallback,
                            reason='Локальный тестовый ответ.')
        return ScriptedModel()

    def prepared(self):
        model = self.model()
        core = Recall(model)
        first = core.decide(self.original)
        exp = core.save_outcome(first['id'], True, .8, 'Тестовый исход')
        return core, model, exp

    def test_blocked_fallback_and_determinism(self):
        snapshots = []
        for _ in range(2):
            core, model, exp = self.prepared()
            result = core.decide(self.current)
            self.assertEqual(result['source'], 'MODEL')
            self.assertEqual(result['action'], self.fallback)
            self.assertEqual(model.calls, 2)  # Initial decision + ordinary fallback only.
            ev = result['evidence']
            self.assertEqual(ev['status'], self.status)
            self.assertIn('недоступен', ev['reason'])
            self.assertNotIn('action', ev)
            self.assertEqual(ev['experience'], exp)
            self.assertGreaterEqual(ev['similarity'], .5)
            self.assertEqual(ev['similarity'], similarity(features_for(self.original), features_for(self.current)))
            self.assertEqual(core.snapshot()['experiences'], [exp])
            snapshots.append(core.snapshot())
        self.assertEqual(*snapshots)

    def test_constraint_scope(self):
        for text in ['Откат недоступен.', 'Недоступен откат версии.',
                     'Откат версии не доступен.', 'Нельзя откатить версию.',
                     'Откат запрещён.']:
            with self.subTest(text=text):
                self.assertIsNotNone(critical_mismatch(self.action, self.original, text))
        for text in ['Доступен откат версии.', 'Журнал недоступен.',
                     'Откат не запрещён.', 'Откат не невозможен.']:
            with self.subTest(text=text):
                self.assertIsNone(critical_mismatch(self.action, self.original, text))
        self.assertIsNotNone(critical_mismatch('Выполнить удаление файла',
                                              'Удаление разрешено.', 'Удаление запрещено.'))
        self.assertIsNone(critical_mismatch(self.action, 'Откат запрещён.', 'Откат запрещён.'))
        self.assertIsNone(critical_mismatch('Выполнить откат версии', '', 'Нельзя выполнить удаление.'))
        self.assertIsNotNone(critical_mismatch('Выполнить откат версии', '', 'Нельзя выполнить откат версии.'))
        self.assertIsNotNone(critical_mismatch('Выполнить откат версии', '', 'Запрещено выполнить откат версии.'))

    def test_fallback_error_preserves_state_and_status(self):
        core, model, _ = self.prepared()
        before = core.snapshot()
        with patch.object(model, 'decide', side_effect=ModelError('Тестовая ошибка')) as call:
            with self.assertRaisesRegex(ModelError, self.status):
                core.decide(self.current)
            self.assertEqual(call.call_count, 1)
        self.assertEqual(core.snapshot(), before)

    def test_demo_flow_valid_then_blocked(self):
        model = self.model()
        app = create_app(model)
        client = app.test_client()
        self.assertEqual(client.get('/').status_code, 200)
        state = next(iter(app.extensions['recall_states'].values()))
        def post(**data):
            response = client.post('/', data=dict(nonce=state.nonce, **data), follow_redirects=True)
            self.assertEqual(response.status_code, 200)
            return response.get_data(as_text=True)
        post(operation='decide', situation=self.original)
        post(operation='outcome', decision_id='1', success='on', score='.8', feedback='Тестовый исход')
        html = post(operation='decide', situation=self.original.replace('Орион', 'Вега'))
        self.assertIn('Откатить версию сервиса Вега до предыдущей', html)
        self.assertIn('No additional model request was needed.', html)
        self.assertNotIn(self.status, html)
        self.assertEqual(model.calls, 1)
        html = post(operation='decide', situation=self.current)
        self.assertIn(self.status, html)
        self.assertIn(self.fallback, html)
        self.assertNotIn('Selected action:', html)
        self.assertNotIn('No additional model request was needed.', html)
        self.assertEqual(model.calls, 2)
        self.assertIn(self.status, client.get('/').get_data(as_text=True))
        post(operation='outcome', decision_id='3', success='on', score='.8', feedback='Новый исход')
        self.assertEqual(state.core.snapshot()['experiences'][-1]['action'], self.fallback)


class EnglishDemoRegression(unittest.TestCase):
    def test_english_demo_flow(self):
        class EnglishModel:
            ready = True
            calls = 0
            def decide(self, text, features):
                self.calls += 1
                action = ('Roll back service Alpha to the previous version' if self.calls == 1
                          else 'Inspect the error logs for service Beta')
                return dict(features=features, action=action, reason='Check the result before taking further action.')
        model = EnglishModel()
        app = create_app(model)
        client = app.test_client()
        client.get('/')
        state = next(iter(app.extensions['recall_states'].values()))
        def post(**data):
            response = client.post('/', data=dict(nonce=state.nonce, **data), follow_redirects=True)
            self.assertEqual(response.status_code, 200)
            html = response.get_data(as_text=True)
            self.assertNotRegex(html, r'[\u0400-\u04ff]')
            return html
        post(operation='decide', situation=SCENARIO['situation_a'])
        post(operation='outcome', decision_id='1', success='on', score='.8', feedback=SCENARIO['outcome']['feedback'])
        original_experience = state.core.snapshot()['experiences'][0]
        html = post(operation='decide', situation=SCENARIO['situation_b'])
        self.assertIn('Roll back service Beta to the previous version', html)
        self.assertEqual(model.calls, 1)
        self.assertEqual(state.core.snapshot()['decisions'][-1]['source'], 'EXPERIENCE')
        blocked = SCENARIO['situation_b'].replace('Rollback is available', 'Rollback is unavailable')
        html = post(operation='decide', situation=blocked)
        self.assertIn('TRANSFER BLOCKED: critical mismatch', html)
        self.assertIn('Inspect the error logs for service Beta', html)
        self.assertNotIn('Selected action:', html)
        self.assertEqual(model.calls, 2)
        result = state.core.snapshot()['decisions'][-1]
        self.assertEqual(result['source'], 'MODEL')
        self.assertEqual(result['evidence']['experience'], original_experience)
        self.assertEqual(result['evidence']['similarity'],
                         similarity(features_for(SCENARIO['situation_a']), features_for(blocked)))
        self.assertEqual(state.core.snapshot()['experiences'], [original_experience])

    def test_english_constraint_scope(self):
        for action, constraint in [
            ('Roll back service Orion', 'Rollback is unavailable'),
            ('Perform a rollback', 'Rollback is not available'),
            ('Roll back service Vega', 'Do not roll back service Vega'),
            ('Restart service Phoenix', 'Restart is forbidden'),
            ('Restart service Phoenix', 'Cannot restart service Phoenix'),
        ]:
            with self.subTest(constraint=constraint):
                self.assertIsNotNone(critical_mismatch(action, '', constraint))
        for constraint in ('Rollback is available', 'Rollback is not forbidden',
                           'Logs are unavailable', 'Do not restart service Beta'):
            self.assertIsNone(critical_mismatch('Roll back service Beta', '', constraint))

    def test_model_requests_english(self):
        with patch.dict(os.environ, {'NEBIUS_API_KEY':'test-only-placeholder', 'NEBIUS_MODEL_ID':'nvidia/test-nemotron'}):
            client = NebiusClient()
            features = features_for(SCENARIO['situation_a'])
            answer = dict(features=features, action='Roll back service Alpha', reason='The issue followed an update.')
            reply = {'choices':[{'finish_reason':'stop', 'message':{'content':canonical(answer)}}]}
            with patch.object(client, '_request', return_value=(reply, 200)) as transport:
                client.decide(SCENARIO['situation_a'], features)
                prompt = transport.call_args.args[1]['messages'][0]['content']
                self.assertIn('Write action and reason in English', prompt)
                self.assertNotIn('in Russian', prompt)



if __name__ == '__main__':
    unittest.main(verbosity=2)
