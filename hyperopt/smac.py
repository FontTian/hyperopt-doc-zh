"""
Routines for interfacing with SMAC


http://www.cs.ubc.ca/labs/beta/Projects/SMAC/

"""
from functools import partial
from pyll.base import Apply
from base import DuplicateLabel


class Cond(object):
    def __init__(self, name, val, op):
        self.op = op
        self.name = name
        self.val = val

    def __str__(self):
        return 'Cond{%s %s %s}' %  (self.name, self.op, self.val)

    def __eq__(self, other):
        return self.op == other.op and self.name == other.name and self.val == other.val

    def __hash__(self):
        return hash((self.op, self.name, self.val))

    def __repr__(self):
        return str(self)

EQ = partial(Cond, op='=')

def expr_to_config(expr, conditions, hps):
    if conditions is None:
        conditions = ()
    assert isinstance(expr, Apply)
    if expr.name == 'switch':
        idx = expr.inputs()[0]
        options = expr.inputs()[1:]
        assert idx.name == 'hyperopt_param'
        assert idx.arg['obj'].name == 'randint'
        expr_to_config(idx, conditions, hps)
        for ii, opt in enumerate(options):
            expr_to_config(opt,
                           conditions + (EQ(idx.arg['label'].obj, ii),),
                           hps)
    elif expr.name == 'hyperopt_param':
        label = expr.arg['label'].obj
        if label in hps:
            if hps[label]['node'] != expr.arg['obj']:
                raise DuplicateLabel(label)
            hps[label]['conditions'].add(conditions)
        else:
            hps[label] = {'node': expr.arg['obj'],
                          'conditions': set((conditions,))}
    else:
        for ii in expr.inputs():
            expr_to_config(ii, conditions, hps)



def test_expr_to_config():
    import hp
    from pyll import as_apply

    z = hp.randint('z', 10)
    a = hp.choice('a',
                  [
                      hp.uniform('b', -1, 1) + z,
                      {'c': 1, 'd': hp.choice('d',
                                              [3 + hp.loguniform('c', 0, 1),
                                               1 + hp.loguniform('e', 0, 1)])
                      }])

    expr = as_apply((a, z))

    hps = {}
    expr_to_config(expr, (True,), hps)

    for label, dct in hps.items():
        print label
        print '  dist: %s(%s)' % (
            dct['node'].name,
            ', '.join(map(str, [ii.eval() for ii in dct['node'].inputs()])))
        if len(dct['conditions']) > 1:
            print '  conditions (OR):'
            for condseq in dct['conditions']:
                print '    ', ' AND '.join(map(str, condseq))
        elif dct['conditions']:
            for condseq in dct['conditions']:
                print '  conditions :', ' AND '.join(map(str, condseq))


    assert hps['a']['node'].name == 'randint'
    assert hps['b']['node'].name == 'uniform'
    assert hps['c']['node'].name == 'loguniform'
    assert hps['d']['node'].name == 'randint'
    assert hps['e']['node'].name == 'loguniform'
    assert hps['z']['node'].name == 'randint'

    assert set([(True, EQ('a', 0))]) == set([(True, EQ('a', 0))])
    assert hps['a']['conditions'] == set([(True,)])
    assert hps['b']['conditions'] == set([
        (True, EQ('a', 0))]), hps['b']['conditions']
    assert hps['c']['conditions'] == set([
        (True, EQ('a', 1), EQ('d', 0))])
    assert hps['d']['conditions'] == set([
        (True, EQ('a', 1))])
    assert hps['e']['conditions'] == set([
        (True, EQ('a', 1), EQ('d', 1))])
    assert hps['z']['conditions'] == set([
        (True,),
        (True, EQ('a', 0))])

