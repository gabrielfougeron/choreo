{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

..    {% block attributes %}
..    {%- if attributes %}
..    .. rubric:: {{ _('Module Attributes') }}
..    .. autosummary::
..       :recursive:
..    {% for item in attributes %}
..       {{ item }}
..    {%- endfor %}
..    {% endif %}
..    {%- endblock %}
.. 
..    {%- block functions %}
..    {%- if functions %}
..    .. rubric:: {{ _('Functions') }}
..    .. autosummary::
..       :recursive:
..    {% for item in functions %}
..       {{ item }}
..    {%- endfor %}
..    {% endif %}
..    {%- endblock %}
.. 
..    {%- block exceptions %}
..    {%- if exceptions %}
..    .. rubric:: {{ _('Exceptions') }}
..    .. autosummary::
..       :recursive:
..    {% for item in exceptions %}
..       {{ item }}
..    {%- endfor %}
..    {% endif %}
..    {%- endblock %}
.. 
..    {%- block modules %}
..    {%- if modules %}
..    .. rubric:: Modules
..    .. autosummary::
..       :recursive:
..    {% for item in modules %}
..       {{ item }}
..    {%- endfor %}
..    {% endif %}
..    {%- endblock %}
