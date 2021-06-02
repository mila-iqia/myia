from myia.abstract import data
from myia.abstract.to_abstract import from_value, type_to_abstract


def test_from_value():
    assert from_value(3) is data.AbstractAtom({"value": 3, "interface": int})
    assert from_value(True) is data.AbstractAtom(
        {"value": True, "interface": bool}
    )
    assert from_value(False) is data.AbstractAtom(
        {"value": False, "interface": bool}
    )
    assert from_value(3.0) is data.AbstractAtom({"interface": float})


def test_from_value_broaden():
    assert from_value(True, broaden=True) is data.AbstractAtom(
        {"interface": bool}
    )
    assert from_value(3, broaden=True) is data.AbstractAtom({"interface": int})


def test_from_value_tuple():
    assert from_value((1, (False, 3.5))) is data.AbstractStructure(
        [
            data.AbstractAtom({"value": 1, "interface": int}),
            data.AbstractStructure(
                [
                    data.AbstractAtom({"value": False, "interface": bool}),
                    data.AbstractAtom({"interface": float}),
                ],
                {"interface": tuple},
            ),
        ],
        {"interface": tuple},
    )
    assert from_value(
        (1, (False, 3.5)), broaden=True
    ) is data.AbstractStructure(
        [
            data.AbstractAtom({"interface": int}),
            data.AbstractStructure(
                [
                    data.AbstractAtom({"interface": bool}),
                    data.AbstractAtom({"interface": float}),
                ],
                {"interface": tuple},
            ),
        ],
        {"interface": tuple},
    )


def test_type_to_abstract():
    assert type_to_abstract(int) is data.AbstractAtom({"interface": int})
    assert from_value(int) is data.AbstractStructure(
        [data.AbstractAtom({"interface": int})], {"interface": type}
    )
