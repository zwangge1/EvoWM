"""Frozen intervention identity, isolation, matching, and loss-gradient checks."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
import test_allen_anatomical as anatomical_fixtures
spec=importlib.util.spec_from_file_location('diagnose_checkpoint',ROOT/'scripts/allen_world/diagnose_v3_checkpoint.py')
diag=importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


class CheckpointDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.fixture=anatomical_fixtures.AnatomicalTests()
        self.fixture.setUp()

    def model(self,**kwargs):
        cfg,harness=self.fixture.build(**kwargs)
        harness.model.eval()
        return cfg,harness.model

    def inputs(self,n=17):
        rng=np.random.default_rng(19)
        return rng.normal(size=(n,3,8)).astype('float32'),rng.normal(size=(n,3,11)).astype('float32')

    def test_trace_and_interventions_independent_of_batching(self):
        _,model=self.model()
        context,actions=self.inputs()
        a=diag.trace_predictions(model,context,actions,4)
        b=diag.trace_predictions(model,context,actions,17)
        for key in a:
            np.testing.assert_allclose(a[key],b[key],atol=2e-6,rtol=2e-6)
        mean=a['predicted_embedding_2'].mean(axis=0)
        p,donors=diag.intervention_predictions(model,a,mean,3,4)
        q,other=diag.intervention_predictions(model,b,mean,3,17)
        for method in p:
            np.testing.assert_allclose(p[method],q[method],atol=2e-6,rtol=2e-6)
        for x,y in zip(donors,other): np.testing.assert_array_equal(x['donor'],y['donor'])
        self.assertGreater(np.max(np.abs(p['zero_rich']-p['baseline'])),1e-8)
        for method,value in p.items():
            total=np.logaddexp.reduce(value[:,1:]*np.log(10),axis=1)/np.log(10)
            np.testing.assert_allclose(total,a['predicted_log_total'],atol=1e-6,rtol=0)

    def test_connection_masks_are_negative_controls(self):
        for connection in ['none','scalar']:
            with self.subTest(connection=connection):
                _,model=self.model(connection=connection)
                context,actions=self.inputs()
                trace=diag.trace_predictions(model,context,actions,17)
                p,_=diag.intervention_predictions(model,trace,trace['predicted_embedding_2'].mean(axis=0),2,17)
                for method in ['zero_rich','mean_rich','shuffle_rich_00','shuffle_rich_01']:
                    np.testing.assert_array_equal(p[method],p['baseline'])
                if connection=='none': np.testing.assert_array_equal(p['zero_upper'],p['baseline'])

    def test_single_has_no_fictitious_upper_interventions(self):
        _,model=self.model(single=True)
        context,actions=self.inputs()
        trace=diag.trace_predictions(model,context,actions,5)
        p,donors=diag.intervention_predictions(model,trace,None,5,5)
        self.assertEqual(list(p),['baseline'])
        self.assertEqual(donors,[])

    def test_permutation_is_bijection_and_stays_within_predicted_size_bins(self):
        values=np.linspace(0,4,101)
        donor,bins=diag.matched_permutation(values,11)
        np.testing.assert_array_equal(np.sort(donor),np.arange(101))
        np.testing.assert_array_equal(bins,bins[donor])
        self.assertTrue(np.all(donor!=np.arange(101)))
        np.testing.assert_array_equal(donor,diag.matched_permutation(values,11)[0])
        with self.assertRaises(ValueError): diag.matched_permutation([np.nan],0)

    def test_identity_rejects_wrong_target_or_observation(self):
        ds=SimpleNamespace(S=np.arange(24,dtype=float).reshape(6,4)/10,
                           table=pd.DataFrame({'tip':list('abcdef'),'family':['A']*6}))
        windows=SimpleNamespace(rows=np.array([[0,1,2,3],[1,2,3,4]]))
        data=pd.DataFrame({'window':[0,1],'target_tip':['d','e'],'source_tip':['c','d'],
                           'family':['A','A'],'context_0_tip':['a','b'],'context_1_tip':['b','c'],
                           'context_2_tip':['c','d'],'complete_three_bones':[True,True]})
        for j,c in enumerate(['m_tarsus','b_fem','b_tbt','b_tmt']): data['observed_'+c]=ds.S[[3,4],j]
        diag.assert_identity(ds,windows,data)
        data.loc[0,'target_tip']='f'
        with self.assertRaises(AssertionError): diag.assert_identity(ds,windows,data)
        data.loc[0,'target_tip']='d';data.loc[0,'observed_b_fem']+=1
        with self.assertRaises(AssertionError): diag.assert_identity(ds,windows,data)

    def test_gradients_only_use_training_data_and_do_not_update_weights(self):
        cfg,model=self.model()
        rng=np.random.default_rng(8)
        ds=SimpleNamespace(S=np.tile(self.fixture.values.numpy(),(3,1)),pix=rng.normal(size=(12,8)).astype('float32'),
                           rows_mask=np.ones(12,dtype=bool),table=pd.DataFrame({'tip':[f's{i}' for i in range(12)]}))
        windows=SimpleNamespace(rows=np.array([[i,i+1,i+2,i+3] for i in range(9)]))
        actions=rng.normal(size=(9,3,11)).astype('float32')
        before={k:v.clone() for k,v in model.state_dict().items()}
        result=diag.gradient_diagnostics(model,cfg,ds,windows,actions,max_per_group=4)
        self.assertEqual(result['status'],'ok')
        self.assertIn('latent_prediction_level2',result['losses'])
        self.assertIn('excludes SIGReg',result['scope'])
        for k,v in model.state_dict().items(): torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        ds.rows_mask[:]=False
        with self.assertRaises(AssertionError): diag.gradient_diagnostics(model,cfg,ds,windows,actions)

    def test_constant_latent_rank_zero(self):
        stats=diag.latent_statistics(np.ones((20,3)))
        self.assertEqual(stats['effective_rank'],0)

    def test_provenance_rejects_modified_input_or_inference_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';checkpoint=root/'object.ckpt';config=root/'config.yaml'
            table=root/'table.csv';meta=root/'table.columns.json';tree=root/'tree.nwk'
            paths={'checkpoint_sha256':checkpoint,'config_sha256':config,'table_sha256':table,
                   'table_metadata_sha256':meta,'tree_sha256':tree}
            for p in paths.values(): p.write_text('original',encoding='utf-8')
            rel='scripts/allen_world/evaluate.py'
            local=root/rel;snapshot=run/'source'/rel
            local.parent.mkdir(parents=True);snapshot.parent.mkdir(parents=True)
            local.write_bytes(b'original\r\n');snapshot.write_bytes(b'original\n')
            manifest={'code_sha256':{rel:diag.file_sha(snapshot)}}
            old={'provenance':{key:diag.file_sha(p) for key,p in paths.items()}}
            cfg={'data':{'dataset':{'species_table':'table.csv','tree_file':'tree.nwk'}}}
            diag.verify_provenance(root,run,checkpoint,config,cfg,old,manifest)
            table.write_text('changed',encoding='utf-8')
            with self.assertRaises(ValueError): diag.verify_provenance(root,run,checkpoint,config,cfg,old,manifest)
            table.write_text('original',encoding='utf-8');local.write_bytes(b'changed\n')
            with self.assertRaises(ValueError): diag.verify_provenance(root,run,checkpoint,config,cfg,old,manifest)


if __name__=='__main__': unittest.main()
