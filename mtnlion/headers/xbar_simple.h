class XBarSimple : public Expression
{
public:
	XBarSimple() : Expression() {}

	void eval(Array<double>& values, const Array<double>& x, const ufc::cell& c) const {
	    xbar->eval(values, x, c);
	}

	std::shared_ptr<GenericFunction> xbar;
};
