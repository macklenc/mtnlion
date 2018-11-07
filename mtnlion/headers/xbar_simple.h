class XBarSimple : public Expression
{
public:
	XBarSimple() : Expression() {}

	void eval(Array<double>& values, const Array<double>& x, const ufc::cell& c) const {
	    xbar->eval(values, x, c);

        std::cout << "Index: " << c.index << std::endl;
		std::cout << "XBarSimple x:" << x << std::endl;
		std::cout << "XBarSimple val:" << values << std::endl;
	}

	std::shared_ptr<GenericFunction> xbar;
};
